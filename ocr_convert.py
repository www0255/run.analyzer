"""截图转 CSV：把跑步 App 的记录截图解析成结构化跑步记录。

两条引擎：
  1. 离线引擎：RapidOCR(onnxruntime) 本地识别，不产生任何网络请求。惰性导入，
     未安装时只抛 OcrDependencyError，绝不影响 app 启动。
  2. 视觉引擎：把图片 base64 交给 OpenAI 兼容 chat.completions，要求只返回 CSV 文本。

解析分两种模式（自动判别）：
  - 卡片模式（真实截图的主模式）：一张图 = 一次跑步的记录详情页。版式是
    「大字号数值 + 下方/左侧中文标签」。用标签锚定：先定位中文标签框，再取与它
    列对齐（数值在上）或同行左侧（数值在左）的数值框。绝不抓页面上的第一个数字。
  - 表格模式（次要模式）：图里确实是多行列表时，按 y 聚行、按 x 排序，用表头
    关键词识别列，再逐行取值。
"""

import base64
import importlib.util
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger("run_analyzer.ocr")

COLUMNS = ["日期", "里程(km)", "时长(分钟)", "配速(min/km)", "平均心率"]
OPTIONAL_COLUMNS = ["平均心率"]

PACE_TOLERANCE = 0.15          # min/km，配速与 时长/里程 的自洽阈值
MIN_PARSE_CONFIDENCE_HITS = 2  # 卡片模式至少要命中的标签数

DISTANCE_LABELS = {"公里", "千米", "里程", "距离", "总里程", "公里数", "距离km"}
DURATION_LABELS = {"用时", "时长", "总时长", "运动时长", "时间", "总用时"}
PACE_LABELS = {"平均配速", "配速"}
HEART_RATE_LABELS = {"平均心率", "心率", "平均心率bpm", "心率bpm"}

BT = chr(96)  # 反引号，避免与外部字符串定界符冲突


class OcrDependencyError(RuntimeError):
    """离线引擎缺依赖时抛出，调用方必须捕获并给出友好提示。"""


@dataclass
class OcrLine:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(self.x1 - self.x0, 1.0)

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 1.0)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass
class ParseResult:
    records: list
    raw_lines: list
    engine: str
    elapsed: float
    error: str = ""
    notes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.records) and not self.error


# --------------------------------------------------------------------------- #
# 引擎
# --------------------------------------------------------------------------- #
_ENGINE = None


def ocr_available() -> bool:
    """只查模块是否可导入，不加载模型（毫秒级，可安全用于 UI 判断）。"""
    try:
        return importlib.util.find_spec("rapidocr_onnxruntime") is not None
    except (ImportError, ValueError):
        return False


def get_engine():
    """惰性构造 RapidOCR 单例；缺依赖或模型加载失败都转成 OcrDependencyError。"""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    try:
        from rapidocr_onnxruntime import RapidOCR  # 惰性导入：装在函数内
    except ImportError as exc:
        raise OcrDependencyError(
            "离线 OCR 依赖导入失败（{0}）：{1} ｜ 若在 Debian/Ubuntu，请确认 packages.txt 的"
            " libgl1 / libglib2.0-0 / libgomp1 已生效".format(type(exc).__name__, str(exc)[:200])
        ) from exc
    try:
        _ENGINE = RapidOCR()
    except Exception as exc:  # noqa: BLE001 - onnxruntime/模型缺失都归为依赖问题
        logger.warning("RapidOCR 初始化失败: error=%s", type(exc).__name__)
        raise OcrDependencyError(
            "离线 OCR 引擎初始化失败（{0}）：{1}".format(type(exc).__name__, str(exc)[:200])
        ) from exc
    logger.info("离线 OCR 引擎已加载: engine=RapidOCR")
    return _ENGINE


def _diag_exc(exc) -> str:
    return ("{0}: {1}".format(type(exc).__name__, exc))[:200]


def engine_diagnostics() -> dict:
    """环境自检：逐项独立 try/except，绝不抛异常，也不发任何网络请求。"""
    import platform, sys
    rows, status = [], {}

    def add(item, detail, ok):
        rows.append({"项目": item, "状态": "可用" if ok else "不可用", "详情": str(detail)[:200]})

    def probe(key, label, func):
        try:
            detail, status[key] = func(), True
        except Exception as exc:  # noqa: BLE001 - 每项独立，一项崩溃不影响其他项
            detail, status[key] = _diag_exc(exc), False
        add(label, detail, status[key])

    def version_of(name):
        return "版本 " + str(getattr(importlib.import_module(name), "__version__", "未知"))

    def rapidocr_class():
        from rapidocr_onnxruntime import RapidOCR  # noqa: F401 - 只验证导入链本身
        return "RapidOCR 类可导入"

    def engine_detail():
        get_engine()  # 失败抛 OcrDependencyError，消息里已带原始异常文本
        return "模型加载成功"

    probe("cv2", "cv2 (OpenCV) 导入", lambda: version_of("cv2"))
    probe("onnxruntime", "onnxruntime 导入", lambda: version_of("onnxruntime"))
    probe("numpy", "numpy 导入", lambda: version_of("numpy"))
    probe("rapidocr", "from rapidocr_onnxruntime import RapidOCR", rapidocr_class)
    probe("engine", "RapidOCR() 构造", engine_detail)
    probe("openai", "openai 导入（大模型识别）", lambda: version_of("openai"))
    probe("platform", "运行平台", lambda: "{0} | Python {1} | Linux={2}".format(
        platform.platform(), sys.version.split()[0], sys.platform.startswith("linux")))
    verdict = "可离线识别" if status["engine"] else ("仅可用大模型识别" if status.get("openai") else "两者都不可用")
    return {"rows": rows, "verdict": verdict}

def decode_image(data: bytes):
    """解码为 BGR ndarray + (宽, 高)；RapidOCR 内部按 OpenCV 约定使用 BGR。"""
    from io import BytesIO

    from PIL import Image

    with Image.open(BytesIO(data)) as image:
        image = image.convert("RGB")
        width, height = image.size
        array = np.asarray(image)
    return np.ascontiguousarray(array[:, :, ::-1]), (width, height)


def _normalize_raw(result):
    """兼容 RapidOCR 各版本的返回结构。"""
    if result is None:
        return []
    if hasattr(result, "txts") and hasattr(result, "boxes"):     # 2.x 结果对象
        boxes = result.boxes if result.boxes is not None else []
        return [[box, text, 1.0] for box, text in zip(boxes, result.txts)]
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], (list, tuple)):
        return list(result[0])                                    # 1.x: (result, elapse)
    if isinstance(result, (list, tuple)):
        return list(result)
    return []


def _box_bounds(box):
    points = np.asarray(box, dtype=float).reshape(-1, 2)
    return float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())


def run_ocr(data: bytes) -> list:
    """离线 OCR：返回带坐标的文本行。网络零请求（纯本地推理）。"""
    engine = get_engine()
    array, (width, height) = decode_image(data)
    started = time.perf_counter()
    raw = engine(array)
    elapsed = time.perf_counter() - started
    lines = []
    for item in _normalize_raw(raw):
        try:
            box, text = item[0], item[1]
        except (TypeError, IndexError):
            continue
        text = str(text).strip()
        if not text:
            continue
        try:
            x0, y0, x1, y1 = _box_bounds(box)
        except Exception:  # noqa: BLE001 - 单个框解析失败不影响整体
            continue
        lines.append(OcrLine(text, x0, y0, x1, y1))
    lines.sort(key=lambda line: (line.y0, line.x0))
    logger.info("离线 OCR 完成: 图片=%dx%d 文本框=%d 耗时=%.2fs", width, height, len(lines), elapsed)
    return lines


# --------------------------------------------------------------------------- #
# 数值宽容解析
# --------------------------------------------------------------------------- #
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    return re.sub(r"\s+", "", text)


def strip_units(text: str) -> str:
    """去掉括号内单位并小写化：'平均心率(bpm)' -> '平均心率'。"""
    text = re.sub(r"[（(\[【][^）)\]】]*[）)\]】]", "", str(text or ""))
    return normalize_text(text).lower()


def parse_distance(text) -> float | None:
    """3.0 / 3,0 / 3km / 3.0公里 / 10.50 -- 返回公里数。"""
    if text is None:
        return None
    raw = normalize_text(text).lower().replace("km", "").replace("公里", "").replace("千米", "")
    match = re.search(r"\d+(?:[.,]\d+)?", raw)
    if not match:
        return None
    token = match.group(0).replace(",", ".")
    try:
        value = float(token)
    except ValueError:
        return None
    return value if 0 < value <= 500 else None


def parse_duration(text) -> float | None:
    """01:12:30 -> 72.5 分；21:30 -> 21.5 分；21 -> 21 分。"""
    if text is None:
        return None
    raw = normalize_text(text).replace("分钟", "").replace("min", "").replace("小时", ":")
    match = re.search(r"\d+(?::\d+){0,2}(?:\.\d+)?", raw)
    if not match:
        return None
    try:
        numbers = [float(part) for part in match.group(0).split(":")]
    except ValueError:
        return None
    if len(numbers) == 3:
        minutes = numbers[0] * 60 + numbers[1] + numbers[2] / 60.0
    elif len(numbers) == 2:
        minutes = numbers[0] + numbers[1] / 60.0
    else:
        minutes = numbers[0]
    return minutes if 0 < minutes <= 24 * 60 else None


_PACE_SPLIT = re.compile(r"^(?P<m>\d{1,2})\s*['\u2032\u2019\u00b4" + BT + r"]\s*(?P<s>\d{1,2})\s*[\"\u2033\u201c\u201d]?")
_PACE_COLON = re.compile(r"^(?P<m>\d{1,2}):(?P<s>\d{2})(?:\.\d+)?$")


def parse_pace(text) -> float | None:
    """6'12\" / 6:12 / 6.2 / 372(秒) -> min/km。"""
    if text is None:
        return None
    raw = re.sub(r"(min/km|/km|每公里|分钟/公里)", "", normalize_text(text))
    match = _PACE_SPLIT.search(raw)
    if match:
        minutes = int(match.group("m")) + int(match.group("s")) / 60.0
        return minutes if 1.5 <= minutes <= 30 else None
    match = _PACE_COLON.search(raw)
    if match:
        minutes = int(match.group("m")) + int(match.group("s")) / 60.0
        return minutes if 1.5 <= minutes <= 30 else None
    match = re.search(r"\d+(?:\.\d+)?", raw.replace(",", "."))
    if not match:
        return None
    token = match.group(0)
    try:
        value = float(token)
    except ValueError:
        return None
    if "." in token:
        return value if 1.5 <= value <= 30 else None
    if 60 <= value <= 1800:                 # 纯数字按秒/公里处理：372 -> 6.2
        value = value / 60.0
    return value if 1.5 <= value <= 30 else None


def parse_heart_rate(text) -> float | None:
    """173 / 173bpm -> 173。"""
    if text is None:
        return None
    match = re.search(r"\d{2,3}", normalize_text(text).lower().replace("bpm", ""))
    if not match:
        return None
    value = float(match.group(0))
    return value if 30 <= value <= 240 else None


def _date_from_text(text: str) -> str | None:
    match = re.search(r"(\d{4})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})", text)
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    if not (2000 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31):
        return None
    return "{0:04d}-{1:02d}-{2:02d}".format(year, month, day)


def _split_time_tail(flat: str) -> list:
    """切开「日期+时间」粘连的尾部时间令牌，返回 [(head, hour, minute)]，第 0 项为首选切法。

    不同 OCR 版本会把 '2026-11-1 07:00' 输出成 '2026-11-107:00'，此时日的末位与
    小时首位无法区分：必须给出两种切法并显式提示，不能静默猜。
    """
    match = re.search(r"(\d{2}):(\d{2})$", flat)
    if match and int(match.group(1)) <= 23:                 # 优先：前导零写法 07:00
        head, hour, minute = flat[:match.start()], match.group(1), match.group(2)
        splits = [(head, hour, minute)]
        base = head.rstrip()
        if len(hour) == 2 and base and base[-1].isdigit():   # 备选：日借回小时首位
            splits.append((base + hour[0], hour[1:], minute))
        return splits
    match = re.search(r"(\d{1,2}):(\d{2})$", flat)
    if match and int(match.group(1)) <= 23:                 # 回退：无前导零的 7:00
        head, hour, minute = flat[:match.start()], match.group(1), match.group(2)
        splits = [(head, hour, minute)]
        base = head.rstrip()
        if base and base[-1].isdigit():                      # 备选：日借一位当小时前导零
            splits.append((base[:-1], "0" + hour, minute))
        return splits
    return []


def parse_date_detail(text) -> tuple:
    """返回 (日期, 歧义提示)。日期取首选切法；两种切法都合法时给出提示。"""
    if not text:
        return None, ""
    flat = unicodedata.normalize("NFKC", str(text))
    splits = _split_time_tail(flat)
    if not splits:
        return _date_from_text(flat), ""
    primary_head, primary_hour, primary_minute = splits[0]
    primary = _date_from_text(primary_head) or _date_from_text(flat)
    if len(splits) > 1:
        alt_head, alt_hour, alt_minute = splits[1]
        alt = _date_from_text(alt_head)
        if primary and alt and alt != primary:
            return primary, "日期存在歧义：{0} {1}:{2} / {3} {4}:{5}，已按前者处理，请核对".format(
                primary, primary_hour, primary_minute, alt, alt_hour, alt_minute
            )
    return primary, ""


def parse_date(text) -> str | None:
    """2026-11-1 / 2026-11-107:00 / 2026/10/23 / 2026年10月18日 -> YYYY-MM-DD。"""
    return parse_date_detail(text)[0]


# --------------------------------------------------------------------------- #
# 标签锚定（卡片模式）
# --------------------------------------------------------------------------- #
_PARSERS = {
    "distance": parse_distance,
    "duration": parse_duration,
    "pace": parse_pace,
    "heart_rate": parse_heart_rate,
}


def label_kind(text: str) -> str | None:
    """严格整词匹配标签，避免 '最快1公里' 被当成里程标签。"""
    cleaned = strip_units(text)
    if cleaned in DISTANCE_LABELS:
        return "distance"
    if cleaned in DURATION_LABELS:
        return "duration"
    if cleaned in PACE_LABELS:
        return "pace"
    if cleaned in HEART_RATE_LABELS:
        return "heart_rate"
    return None


def _candidates_above(label: OcrLine, lines: Iterable[OcrLine]) -> list:
    scored = []
    for line in lines:
        gap = label.y0 - line.y1
        if gap < -2 or gap > max(2.2 * label.height, 40):
            continue
        column_shift = abs(line.cx - label.cx) / label.width
        if column_shift > 1.0:
            continue
        scored.append((gap + 60.0 * column_shift, line))
    return [line for _, line in sorted(scored, key=lambda pair: pair[0])]


def _candidates_left(label: OcrLine, lines: Iterable[OcrLine]) -> list:
    """同行左侧（'10.50 公里' 这种横排版式）：数值中心在标签中心左侧即可。"""
    scored = []
    for line in lines:
        if line.cx >= label.cx:
            continue
        overlap = min(line.y1, label.y1) - max(line.y0, label.y0)
        if overlap < 0.5 * min(line.height, label.height):
            continue
        gap = label.x0 - line.x1
        if gap < -0.8 * label.width:       # 允许与标签轻微横向重叠
            continue
        scored.append((abs(gap), line))
    return [line for _, line in sorted(scored, key=lambda pair: pair[0])]


def find_value(label: OcrLine, kind: str, lines: list) -> str | None:
    """在标签的'上方列对齐'或'同行左侧'找第一个能解析成目标类型的数值框。"""
    parser = _PARSERS[kind]
    order = (_candidates_left, _candidates_above) if kind == "distance" else (_candidates_above, _candidates_left)
    for picker in order:
        for candidate in picker(label, lines):
            if not re.search(r"\d", candidate.text):
                continue
            if parser(candidate.text) is not None:
                return candidate.text
    return None


def _fallback_values(lines: list) -> dict:
    """标签锚定全失败时的保底：图上唯一的版式化数值。"""
    found = {}
    for line in lines:
        text = normalize_text(line.text)
        if "distance" not in found and re.fullmatch(r"\d+(?:[.,]\d+)?(?:公里|km|千米)", text, re.IGNORECASE):
            found["distance"] = text
        if "duration" not in found and re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", text):
            found["duration"] = text
        if "pace" not in found and re.fullmatch(r"\d{1,2}['\u2032\u2019]\d{1,2}[\"\u2033]?", text):
            found["pace"] = text
    return found


def extract_card_record(lines: list) -> dict:
    """一张图 = 一次跑步：标签锚定取值，缺失字段留空。"""
    labels = []
    for line in lines:
        kind = label_kind(line.text)
        if kind:
            labels.append((kind, line))
    hits = {kind for kind, _ in labels}

    picked = {}
    for kind in ("distance", "duration", "pace", "heart_rate"):
        for current_kind, label in labels:
            if current_kind != kind or kind in picked:
                continue
            raw = find_value(label, kind, lines)
            if raw is not None:
                picked[kind] = raw

    for kind, raw in _fallback_values(lines).items():
        picked.setdefault(kind, raw)

    if len(hits) < MIN_PARSE_CONFIDENCE_HITS and not picked:
        return {}

    date, date_note = None, ""
    for line in lines:
        date, date_note = parse_date_detail(line.text)
        if date:
            break

    return {
        "日期": date,
        "里程(km)": parse_distance(picked.get("distance")),
        "时长(分钟)": parse_duration(picked.get("duration")),
        "配速(min/km)": parse_pace(picked.get("pace")),
        "平均心率": parse_heart_rate(picked.get("heart_rate")),
        "_raw": picked,
        "_date_note": date_note,
    }


# --------------------------------------------------------------------------- #
# 表格模式（次要）
# --------------------------------------------------------------------------- #
HEADER_KEYWORDS = {
    "日期": ("日期", "date"),
    "里程(km)": ("里程", "距离", "公里", "km", "distance"),
    "时长(分钟)": ("时长", "用时", "duration"),
    "配速(min/km)": ("配速", "pace"),
    "平均心率": ("心率", "heartrate", "bpm"),
}


def group_rows(lines: list, tolerance: float = 0.6) -> list:
    rows, current = [], []
    for line in sorted(lines, key=lambda item: item.cy):
        if not current or abs(line.cy - current[0].cy) <= tolerance * max(line.height, current[0].height):
            current.append(line)
        else:
            rows.append(current)
            current = [line]
    if current:
        rows.append(current)
    return [sorted(row, key=lambda item: item.x0) for row in rows]


def _cell_kind(text: str) -> str | None:
    cleaned = strip_units(text)
    for kind, words in HEADER_KEYWORDS.items():
        for word in words:
            if word in cleaned:
                return kind
    return None


def extract_table_records(lines: list) -> list:
    rows = group_rows(lines)
    header_row, header_map = None, {}
    for row in rows:
        mapping = {}
        for line in row:
            kind = _cell_kind(line.text)
            if kind and kind not in mapping:
                mapping[kind] = line.cx
        if len(mapping) >= 3:
            header_row, header_map = row, mapping
            break
    if header_row is None:
        return []

    records = []
    for row in rows:
        if row is header_row or len(row) < 2:
            continue
        record = {}
        for line in row:
            if not re.search(r"\d", line.text):
                continue
            nearest = min(header_map, key=lambda kind: abs(header_map[kind] - line.cx))
            if abs(header_map[nearest] - line.cx) > 220 or nearest in record:
                continue
            record[nearest] = line.text
        if len(record) < 2:
            continue
        row_text = " ".join(line.text for line in row)
        records.append({
            "日期": parse_date(record.get("日期")) or parse_date(row_text),
            "里程(km)": parse_distance(record.get("里程(km)")),
            "时长(分钟)": parse_duration(record.get("时长(分钟)")),
            "配速(min/km)": parse_pace(record.get("配速(min/km)")),
            "平均心率": parse_heart_rate(record.get("平均心率")),
            "_raw": record,
        })
    return records


# --------------------------------------------------------------------------- #
# 补全与校验
# --------------------------------------------------------------------------- #
def complete_record(record: dict) -> tuple:
    """缺失字段互算补全；返回 (记录, 提示列表)。"""
    notes = list(record.get("_notes") or [])
    distance = record.get("里程(km)")
    duration = record.get("时长(分钟)")
    pace = record.get("配速(min/km)")

    if distance and pace and not duration:
        record["时长(分钟)"] = round(pace * distance, 4)
        notes.append("时长缺失，已用 配速×里程 推算")
    elif distance and duration and not pace:
        record["配速(min/km)"] = round(duration / distance, 4)
        notes.append("配速缺失，已用 时长÷里程 推算")

    if distance and record.get("时长(分钟)") and record.get("配速(min/km)"):
        expected = record["时长(分钟)"] / distance
        if abs(expected - record["配速(min/km)"]) > PACE_TOLERANCE:
            notes.append(
                "配速 {0:.2f} 与 时长/里程 推算值 {1:.2f} 不一致，识别可能有误，请核对".format(
                    record["配速(min/km)"], expected
                )
            )
    record["_notes"] = notes
    return record, notes


def record_is_valid(record: dict) -> bool:
    distance = record.get("里程(km)")
    return bool(distance) and bool(record.get("配速(min/km)") or record.get("时长(分钟)"))


# --------------------------------------------------------------------------- #
# 对外主入口
# --------------------------------------------------------------------------- #
def extract_records(data: bytes, source_name: str = "") -> ParseResult:
    """单张图片 -> 记录列表（卡片模式 1 条，表格模式 N 条）。异常不外抛。"""
    started = time.perf_counter()
    try:
        lines = run_ocr(data)
    except OcrDependencyError as exc:
        logger.warning("离线引擎不可用: source=%s error=%s", source_name, type(exc).__name__)
        return ParseResult([], [], "离线", time.perf_counter() - started, str(exc))
    except Exception as exc:  # noqa: BLE001 - 识别失败必须降级成提示
        logger.warning("离线识别失败: source=%s error=%s detail=%s", source_name, type(exc).__name__, exc)
        return ParseResult([], [], "离线", time.perf_counter() - started,
                           "图片识别失败（{0}），请换一张更清晰的截图".format(type(exc).__name__))

    raw_lines = [line.text for line in lines]
    card = extract_card_record(lines)
    card_used = bool(card) and record_is_valid(card)
    if card_used:
        records = [complete_record(card)[0]]
    else:
        records = [complete_record(record)[0] for record in extract_table_records(lines)]

    elapsed = time.perf_counter() - started
    logger.info(
        "截图解析完成: source=%s 模式=%s 文本行=%d 有效记录=%d 耗时=%.2fs",
        source_name, "卡片" if card_used else "表格", len(lines), len(records), elapsed,
    )
    notes = []
    for record in records:
        note = record.pop("_date_note", "")
        if note:
            notes.append(note)
    for note in notes:
        logger.warning("解析提示: source=%s note=%s", source_name, note)
    error = "" if records else "未能从这张截图识别出里程/配速/时长，请手动补或换更清晰的截图"
    return ParseResult(records, raw_lines, "离线", elapsed, error, notes)


def records_to_frame(records: list) -> pd.DataFrame:
    rows = [{column: record.get(column) for column in COLUMNS} for record in records]
    if not rows:
        rows = [{column: None for column in COLUMNS}]
    frame = pd.DataFrame(rows, columns=COLUMNS)
    for column in ["里程(km)", "时长(分钟)", "配速(min/km)", "平均心率"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def collect_notes(records: list) -> list:
    notes = []
    for index, record in enumerate(records, start=1):
        for note in record.get("_notes") or []:
            notes.append("第 {0} 条：{1}".format(index, note))
    return notes


def frame_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """UTF-8 带 BOM，Excel 双击直接可开。"""
    return frame.to_csv(index=False).encode("utf-8-sig")


# --------------------------------------------------------------------------- #
# 视觉引擎（OpenAI 兼容）
# --------------------------------------------------------------------------- #
VISION_PROMPT = (
    "这是一张跑步 App 的运动记录截图。请只输出 CSV 文本，不要任何解释、不要代码块标记。\n"
    "第一行必须是表头：日期,里程(km),时长(分钟),配速(min/km),平均心率\n"
    "每行一条记录；日期用 YYYY-MM-DD；时长单位分钟（可用小数）；配速单位 min/km（可用小数）；"
    "心率只填数字。截图上没有的字段留空。"
)


def _parse_vision_csv(text: str) -> pd.DataFrame:
    from io import StringIO

    cleaned = text.replace(BT + BT + BT, "").replace("csv\n", "", 1) if text.lstrip().startswith(BT) else text
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    header_index = next((i for i, line in enumerate(lines) if "日期" in line and "里程" in line), None)
    if header_index is None:
        raise ValueError("模型返回内容里找不到表头（日期/里程），已放弃这次结果")
    kept = [lines[header_index]]
    for line in lines[header_index + 1:]:
        if "," not in line:          # 表头之后的解释性文字/围栏，直接截断
            break
        kept.append(line)
    frame = pd.read_csv(StringIO("\n".join(kept)))
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("模型返回缺少列：" + "、".join(missing))
    return frame[COLUMNS]


def vision_extract(data: bytes, base_url: str, model_name: str, api_key: str) -> tuple:
    """返回 (DataFrame, 错误信息)；绝不外抛，也绝不打印 Key。"""
    if not api_key:
        return None, "未填写 API Key，无法使用大模型识别"
    try:
        from openai import OpenAI  # 延迟导入
    except ImportError:
        return None, "未安装 openai 依赖，请执行 pip install -r requirements.txt"

    encoded = base64.b64encode(data).decode("ascii")
    started = time.perf_counter()
    logger.info("视觉识别开始: base_url=%s model=%s", base_url, model_name)
    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}},
                ],
            }],
            timeout=60.0,
        )
        text = (response.choices[0].message.content or "").strip()
        frame = _parse_vision_csv(text)
    except Exception as exc:  # noqa: BLE001 - 视觉引擎失败必须降级，不许崩
        logger.warning("视觉识别失败: error=%s detail=%s", type(exc).__name__, str(exc)[:200])
        return None, "大模型识别失败（{0}）：{1}".format(type(exc).__name__, str(exc)[:120])
    logger.info("视觉识别成功: model=%s 行数=%d 耗时=%.2fs", model_name, len(frame), time.perf_counter() - started)
    return frame, ""
