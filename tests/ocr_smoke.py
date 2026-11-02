"""截图转 CSV 的自测脚本（不依赖 Streamlit）。

运行：python tests/ocr_smoke.py
1) 数值宽容解析单测；2) 用真实截图的坐标版式做标签锚定单测（不跑 OCR）；
3) 有 samples/ 真实截图时跑一次离线 OCR，与人工读图的 oracle 逐项比对。
"""

import base64
import logging
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ocr_convert as ocr  # noqa: E402

SAMPLE_DIR = ROOT / "samples"

# 人工读图得到的 oracle：(文件, 日期, 里程, 时长分钟, 配速, 心率)
# 含真实跑步记录（属个人数据），因此**不进公开仓库**：从 samples/oracle.json 读取，
# 该文件位于被 .gitignore 排除的 samples/ 下。没有该文件时自动跳过真实图比对。
def load_oracle():
    path = SAMPLE_DIR / "oracle.json"
    if not path.exists():
        return []
    import json
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [(r["file"], r["date"], r["km"], r["min"], r["pace"], r["hr"]) for r in rows]


ORACLE = load_oracle()

FAILED = []


def check(name, actual, expected, tol=0.0):
    if expected is None or actual is None:
        ok = actual == expected
    elif isinstance(expected, str):
        ok = actual == expected
    else:
        ok = abs(float(actual) - float(expected)) <= max(tol, 1e-9)
    print("  [{0}] {1}: {2!r} (期望 {3!r})".format("OK" if ok else "FAIL", name, actual, expected))
    if not ok:
        FAILED.append(name)


def test_parsers():
    print("[1] 数值宽容解析")
    for text, expected in [("6'12\"", 6.2), ("6:12", 6.2), ("6.2", 6.2), ("372", 6.2), ("6", 6.0),
                           ("6'12\"", 6.2), ("6\u203212\u2033", 6.2), ("5'30\"", 5.5)]:
        check("配速 " + text, ocr.parse_pace(text), expected, tol=0.02)
    for text, expected in [("21", 21.0), ("21:30", 21.5), ("0:21:30", 21.5), ("01:12:30", 72.5),
                           ("01:25:07", 85.1167), ("01:30:44", 90.7333)]:
        check("时长 " + text, ocr.parse_duration(text), expected, tol=0.02)
    for text, expected in [("3.0", 3.0), ("3,0", 3.0), ("3km", 3.0), ("3.0公里", 3.0), ("10.50", 10.50)]:
        check("里程 " + text, ocr.parse_distance(text), expected, tol=0.001)
    for text, expected in [("150", 150.0), ("平均心率(bpm)", None), ("150bpm", 150.0)]:
        check("心率 " + text, ocr.parse_heart_rate(text), expected)
    # 以下全是纯函数用例：不加载 OCR 引擎，因此任何 rapidocr 版本（含未安装）结果一致
    for text, expected in [("2026-11-1 07:00", "2026-11-01"), ("2026/03/05", "2026-03-05"),
                           ("2026年4月8日", "2026-04-08"),
                           ("2026-11-107:00", "2026-11-01"),      # rapidocr 1.4.4 的粘连输出
                           ("2026-11-10 7:00", "2026-11-10"),     # 真 10 日 + 无前导零时间
                           ("2026-11-108:30", "2026-11-01")]:     # 粘连：优先 01 日 + 08:30
        check("日期 " + text, ocr.parse_date(text), expected)
    # 粘连写法必须带歧义提示，且提示要写明两种读法（日期取首选切法）
    for text, fragment in [("2026-11-107:00", "2026-11-01 07:00 / 2026-11-10 7:00"),
                           ("2026-11-10 7:00", "2026-11-10 7:00 / 2026-11-01 07:00"),
                           ("2026-11-108:30", "2026-11-01 08:30 / 2026-11-10 8:30")]:
        check("歧义提示 " + text, fragment in ocr.parse_date_detail(text)[1], True)
    for text in ("2026/03/05", "2026-03-05 17:24", "2026-04-08 07:04"):
        check("无歧义不应提示 " + text, ocr.parse_date_detail(text)[1], "")


def test_label_anchoring():
    """按第 1 张截图的真实版式构造坐标，验证'标签锚定'不会抓错数字。"""
    print("[2] 标签锚定（含干扰项：公里点 / 步频 / 最快1公里）")
    L = ocr.OcrLine
    lines = [
        L("G323", 430, 100, 500, 130), L("25C", 380, 145, 430, 175), L("68%", 480, 145, 540, 175),
        L("21", 480, 320, 510, 350), L("19", 700, 415, 730, 445), L("17", 745, 415, 775, 445),
        L("13", 640, 700, 670, 730), L("2026-01-15 07:00", 590, 1040, 810, 1075),
        L("10.50", 60, 1030, 300, 1105), L("公里", 285, 1075, 355, 1110),
        L("00:52:30", 60, 1205, 260, 1260), L("5'00\"", 380, 1205, 510, 1260), L("150", 665, 1205, 755, 1260),
        L("用时", 130, 1270, 210, 1305), L("平均配速", 370, 1270, 520, 1305), L("平均心率(bpm)", 620, 1270, 800, 1305),
        L("186", 130, 1385, 215, 1440), L("119", 405, 1385, 490, 1440), L("63", 690, 1385, 750, 1440),
        L("平均步频", 105, 1450, 240, 1485), L("平均步幅(cm)", 375, 1450, 520, 1485), L("累计爬升(m)", 645, 1450, 790, 1485),
        L("4'17\"", 95, 1565, 200, 1620), L("18017", 390, 1565, 510, 1620), L("1577", 680, 1565, 780, 1620),
        L("最快1公里", 80, 1630, 215, 1665), L("总步数", 410, 1630, 495, 1665), L("大卡", 690, 1630, 760, 1665),
        L("分享运动记录", 300, 1790, 560, 1830),
    ]
    # 模拟 rapidocr 1.4.4 把「2026-11-1 07:00」粘连成「2026-11-107:00」
    glued = [ocr.OcrLine("2026-11-107:00", 590, 1040, 810, 1075) if line.text.startswith("2026-01-15") else line
             for line in lines]
    glued_record = ocr.extract_card_record(glued)
    check("1.4.4 粘连日期不被吃成 02-10", glued_record.get("日期"), "2026-11-01")
    check("粘连时给出歧义提示", bool(glued_record.get("_date_note")), True)
    print("  歧义提示原文:", glued_record.get("_date_note"))

    record = ocr.extract_card_record(lines)
    check("对齐后里程", record.get("里程(km)"), 10.50, tol=0.001)
    check("对齐后时长", record.get("时长(分钟)"), 52.5, tol=0.02)
    check("对齐后配速", record.get("配速(min/km)"), 5.0, tol=0.02)
    check("对齐后心率", record.get("平均心率"), 150.0)
    check("对齐后日期", record.get("日期"), "2026-01-15")
    record, notes = ocr.complete_record(record)
    print("  一致性提示:", notes or "无（自洽）")
    if notes:
        FAILED.append("锚定结果不自洽")


def test_real_images():
    images = sorted(SAMPLE_DIR.glob("*.jpg")) if SAMPLE_DIR.exists() else []
    if not images:
        print("[3] samples/ 无真实截图，跳过高精度验证（准确率未验证）")
        return
    print("[3] 真实截图离线 OCR（共 {0} 张）".format(len(images)))
    if not ocr.ocr_available():
        print("  SKIP: rapidocr_onnxruntime 未安装")
        return
    records = []
    for path in images:
        data = path.read_bytes()
        result = ocr.extract_records(data, path.name)
        print("  --- {0} : 引擎={1} 耗时={2:.2f}s 文本行={3} 记录={4} error={5!r}".format(
            path.name, result.engine, result.elapsed, len(result.raw_lines), len(result.records), result.error))
        print("      原始文本前 10 行:", result.raw_lines[:10])
        if result.records:
            records.append(result.records[0])

    expected_map = {name: (date, dist, dur, pace, hr) for name, date, dist, dur, pace, hr in ORACLE}
    print("[4] 与人工读图 oracle 逐项比对")
    for path, record in zip(images, records):
        oracle = expected_map.get(path.name)
        if not oracle:
            continue
        date, dist, dur, pace, hr = oracle
        print("  " + path.name)
        check("日期", record.get("日期"), date)
        check("里程", record.get("里程(km)"), dist, tol=0.02)
        check("时长", record.get("时长(分钟)"), dur, tol=0.02)
        check("配速", record.get("配速(min/km)"), pace, tol=0.02)
        check("心率", record.get("平均心率"), hr, tol=0.5)

    frame = ocr.records_to_frame(records)
    print("[5] 多图合并结果表（{0} 行）".format(len(frame)))
    print(frame.to_string(index=False))
    print("[6] CSV 导出前 3 字节:", ocr.frame_to_csv_bytes(frame)[:3])


def test_offline_no_socket():
    """屏蔽 socket 后重建引擎并识别：证明离线路径不发任何网络请求。"""
    print("[0] 断网自证：屏蔽 socket 后仍能识别（含模型加载）")
    import socket

    originals = (socket.socket, socket.create_connection)

    def deny(*args, **kwargs):
        raise RuntimeError("network disabled for offline test")

    images = sorted(SAMPLE_DIR.glob("*.jpg"))
    if not images or not ocr.ocr_available():
        print("  SKIP: 无截图或未安装 rapidocr")
        return
    socket.socket = deny
    socket.create_connection = deny
    try:
        ocr._ENGINE = None                      # 强制在断网状态下重建引擎
        result = ocr.extract_records(images[0].read_bytes(), images[0].name)
        check("断网下识别记录数", len(result.records), 1)
        check("断网下里程", result.records[0].get("里程(km)") if result.records else None, ORACLE[0][2] if ORACLE else None, tol=0.02)
    finally:
        socket.socket, socket.create_connection = originals


def test_app_multi_image():
    """应用层多图合并：3 张真实截图 + 1 张损坏图，验证单张失败不影响整批。"""
    print("[7] 应用层多图合并（app.convert_screenshots，含 1 张损坏图）")
    import app

    sample_images = sorted(SAMPLE_DIR.glob("*.jpg"))
    if not sample_images or not ocr.ocr_available():
        print("  SKIP: samples/ 无截图或未安装 rapidocr")
        return
    payloads = [(path.name, path.read_bytes()) for path in sample_images]
    payloads.append(("broken.jpg", b"this-is-not-an-image"))
    frame, notes, failures = app.convert_screenshots(payloads)
    check("合并后行数", len(frame), len(sample_images))
    check("失败提示条数", len(failures), 1)
    for failure in failures:
        print("  失败提示:", failure)
    print(frame.to_string(index=False))


def test_diagnostics():
    """环境自检必须在任何依赖状态下都不抛异常（云端缺 libGL 时靠它自证原因）。"""
    print("[8] 环境自检 engine_diagnostics（不抛异常 + 表格形状）")
    diagnostics = ocr.engine_diagnostics()
    rows = diagnostics["rows"]
    check("自检行数", len(rows), 7)
    check("每行只有 项目/状态/详情 且详情≤200字",
          all(set(row) == {"项目", "状态", "详情"} and len(row["详情"]) <= 200 for row in rows), True)
    check("结论在合法取值内",
          diagnostics["verdict"] in ("可离线识别", "仅可用大模型识别", "两者都不可用"), True)
    for row in rows:
        print("  {0} | {1} | {2}".format(row["项目"], row["状态"], row["详情"]))



# --------------------------------------------------------------------------- #
# AI 接入：Secrets 零配置 / 服务商预设 / 视觉能力守卫（全部离线可测）
# --------------------------------------------------------------------------- #
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8AAAwAB/AF+3w5rAAAAAElFTkSuQmCC"
)
FAKE_SECRET_KEY = "sk-test"        # 假 key：只用来证明它不会出现在页面和日志里
FAKE_MANUAL_KEY = "sk-manual-fake"  # 另一个假 key：证明切预设不清空手填 key
SECRET_STUB = {
    "AI_BASE_URL": "https://api.deepseek.com",
    "AI_MODEL": "deepseek-flash",
    "VISION_MODEL": "deepseek-flash",
    "AI_API_KEY": FAKE_SECRET_KEY,
}


class CaptureHandler(logging.Handler):
    """把日志消息收集起来，供「key 不进日志」的断言使用。"""

    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    def emit(self, record):
        self.sink.append(record.getMessage())


def capture_logs():
    sink = []
    handler = CaptureHandler(sink)
    logging.getLogger().addHandler(handler)
    return sink, handler


def stop_capture(handler):
    logging.getLogger().removeHandler(handler)


def sidebar_inputs(at):
    return {widget.key: widget for widget in at.sidebar.text_input}


def sidebar_preset(at, app):
    return {widget.key: widget for widget in at.sidebar.selectbox}[app.WIDGET_PRESET]


def page_text(at) -> str:
    """把页面上所有文字（含侧边栏控件标签/取值）拼起来，用来查 key 有没有泄漏。"""
    parts = []
    for block in (at, at.sidebar):
        for kind in ("markdown", "caption", "info", "warning", "success", "error", "title",
                     "subheader", "header", "text", "button", "selectbox", "text_input",
                     "radio", "checkbox"):
            for element in getattr(block, kind, []) or []:
                parts.append(str(getattr(element, "value", "")))
                parts.append(str(getattr(element, "label", "")))
    return "\n".join(parts)


def test_vision_payload():
    """payload 函数级证明：只用假的 openai 模块，一次网络请求都不发。"""
    print("[9] 视觉 payload 结构（png/jpg + 独立 vision_model）")
    content = ocr.build_vision_messages(TINY_PNG)[0]["content"]
    url = content[1]["image_url"]["url"]
    check("messages 角色", [m["role"] for m in ocr.build_vision_messages(TINY_PNG)] == ["user"], True)
    check("第一段是 text 提示", content[0]["type"], "text")
    check("第二段是 image_url", content[1]["type"], "image_url")
    check("png 的 data URL 前缀", url.startswith("data:image/png;base64,"), True)
    jpg_like = b"\xff\xd8\xff" + b"\x00" * 32
    check("jpg 的 data URL 前缀",
          ocr.build_vision_messages(jpg_like)[0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,"),
          True)
    print("  payload 结构: {\"role\": \"user\", \"content\": ["
          "{\"type\": \"text\", \"text\": <VISION_PROMPT>}, "
          "{\"type\": \"image_url\", \"image_url\": {\"url\": \"" + url[:44] + "...\"}}]}"
          "（base64 总长度 " + str(len(url)) + "）")

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)

            class Msg:
                content = "日期,里程(km),时长(分钟),配速(min/km),平均心率\n2026-11-01,10.5,52.5,5.0,150\n"

            class Choice:
                message = Msg()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = FakeOpenAI
    original = sys.modules.get("openai")
    sys.modules["openai"] = fake_module
    try:
        frame, error = ocr.vision_extract(TINY_PNG, "https://api.deepseek.com", "deepseek-flash", FAKE_SECRET_KEY)
    finally:
        if original is not None:
            sys.modules["openai"] = original
        else:
            sys.modules.pop("openai", None)
    check("视觉调用无错误", error, "")
    check("解析出 1 行记录", 0 if frame is None else len(frame), 1)
    check("请求用的模型是独立 vision_model", captured.get("model"), "deepseek-flash")
    check("client 用传入 base_url", captured["client"].get("base_url"), "https://api.deepseek.com")
    check("请求体里仍是 png data URL",
          captured["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"), True)
    check("请求体里不含 key 明文", FAKE_SECRET_KEY in str(captured.get("messages")), False)


def test_preset_switch():
    """服务商预设：DeepSeek 填三个值；自定义不回退；key 不被清空。"""
    print("[10] 服务商预设切换（页面级）")
    import app
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    check("初始无异常", [str(e) for e in at.exception] == [], True)
    check("无 secrets 时预设为「自定义」",
          {w.key: w for w in at.sidebar.selectbox}[app.WIDGET_PRESET].value, app.PRESET_CUSTOM)
    sidebar_inputs(at)[app.WIDGET_API_KEY].set_value(FAKE_MANUAL_KEY)
    at.run()
    sidebar_preset(at, app).set_value(app.PRESET_DEEPSEEK)
    at.run()
    inputs = sidebar_inputs(at)
    check("预设后 base_url", inputs[app.WIDGET_BASE_URL].value, "https://api.deepseek.com")
    check("预设后文本模型", inputs[app.WIDGET_MODEL].value, "deepseek-flash")
    check("预设后视觉模型", inputs[app.WIDGET_VISION_MODEL].value, "deepseek-flash")
    check("预设不清空手填 key", inputs[app.WIDGET_API_KEY].value, FAKE_MANUAL_KEY)
    sidebar_preset(at, app).set_value(app.PRESET_CUSTOM)
    at.run()
    inputs = sidebar_inputs(at)
    check("自定义不回退 base_url", inputs[app.WIDGET_BASE_URL].value, "https://api.deepseek.com")
    check("自定义不回退文本模型", inputs[app.WIDGET_MODEL].value, "deepseek-flash")
    check("自定义不回退视觉模型", inputs[app.WIDGET_VISION_MODEL].value, "deepseek-flash")
    check("自定义仍保留手填 key", inputs[app.WIDGET_API_KEY].value, FAKE_MANUAL_KEY)
    check("切换后无异常", [str(e) for e in at.exception] == [], True)


def test_secrets_config():
    """优先级链条 + 守卫函数；最后把假 secrets 真正注入 st.secrets 供页面用例使用。"""
    print("[11] Secrets 优先级与视觉守卫（函数级）")
    import app

    config = app.load_ai_config(SECRET_STUB)
    check("AI_BASE_URL", config["base_url"], "https://api.deepseek.com")
    check("AI_MODEL", config["model_name"], "deepseek-flash")
    check("VISION_MODEL", config["vision_model"], "deepseek-flash")
    check("AI_API_KEY 已读取", bool(config["api_key"]), True)
    check("key 来源标记", config["key_source"], "AI_API_KEY")
    check("回退 DEEPSEEK_API_KEY", app.load_ai_config({"DEEPSEEK_API_KEY": "sk-d"})["key_source"], "DEEPSEEK_API_KEY")
    check("回退 OPENAI_API_KEY", app.load_ai_config({"OPENAI_API_KEY": "sk-o"})["key_source"], "OPENAI_API_KEY")
    empty = app.load_ai_config({})
    check("缺省 base_url", empty["base_url"], "https://api.openai.com/v1")
    check("缺省文本模型", empty["model_name"], "gpt-4o-mini")
    check("视觉模型回退文本模型", empty["vision_model"], "gpt-4o-mini")
    check("无 key 时不算检测到", empty["has_secret_key"], False)

    check("守卫拦 deepseek-chat", "不支持图片输入" in app.vision_guard_message("deepseek-chat"), True)
    check("守卫拦 deepseek-reasoner", "不支持图片输入" in app.vision_guard_message("deepseek-reasoner"), True)
    check("守卫拦 gpt-3.5", "不支持图片输入" in app.vision_guard_message("gpt-3.5-turbo"), True)
    check("守卫提示点名视觉模型", app.DEEPSEEK_VISION_HINT in app.vision_guard_message("deepseek-chat"), True)
    check("守卫放行 gpt-4o-mini", app.vision_guard_message("gpt-4o-mini"), "")
    check("守卫放行视觉模型", app.vision_guard_message("deepseek-flash"), "")

    import streamlit as st
    st.secrets.merge_programmatic_secrets(SECRET_STUB)   # 不落盘：不碰 .streamlit/
    live = app.load_ai_config()                          # 走真实的 st.secrets 读取路径
    check("从 st.secrets 读到 base_url", live["base_url"], SECRET_STUB["AI_BASE_URL"])
    check("从 st.secrets 读到视觉模型", live["vision_model"], SECRET_STUB["VISION_MODEL"])
    check("从 st.secrets 读到 key", bool(live["api_key"]), True)


def test_secrets_page():
    print("[12] 页面：Secrets 检测提示 + 预设生效 + 全页面/日志不出现 key")
    import app
    from streamlit.testing.v1 import AppTest

    logs, handler = capture_logs()
    try:
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    finally:
        stop_capture(handler)
    check("页面无异常", [str(e) for e in at.exception] == [], True)
    texts = [str(c.value) for c in at.caption] + [str(m.value) for m in at.markdown]
    expected = "已从 Secrets 检测到配置（服务商: DeepSeek / 模型: deepseek-flash）"
    check("侧边栏出现 Secrets 检测提示", any(expected in text for text in texts), True)
    inputs = sidebar_inputs(at)
    check("输入框用 secrets 的 base_url", inputs[app.WIDGET_BASE_URL].value, SECRET_STUB["AI_BASE_URL"])
    check("输入框用 secrets 的文本模型", inputs[app.WIDGET_MODEL].value, SECRET_STUB["AI_MODEL"])
    check("输入框用 secrets 的视觉模型", inputs[app.WIDGET_VISION_MODEL].value, SECRET_STUB["VISION_MODEL"])
    joined = page_text(at)
    check("页面任何文字都不含 key", FAKE_SECRET_KEY in joined, False)
    check("日志任何一行都不含 key", any(FAKE_SECRET_KEY in line for line in logs), False)
    print("  检测提示原文:", [text for text in texts if "已从 Secrets" in text])
    print("  页面文字片段不含 key：检查了 {0} 个字符；日志 {1} 行".format(len(joined), len(logs)))


def test_vision_guard_page():
    print("[13] 页面：视觉模型=deepseek-chat 点识别 -> 只提示、零请求")
    import app
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    images = sorted(SAMPLE_DIR.glob("*.jpg"))
    if not images:
        print("  SKIP: samples/ 无截图")
        return

    class FakeUpload:
        def __init__(self, path):
            self.name = path.name
            self._data = path.read_bytes()

        def getvalue(self):
            return self._data

    calls = []
    original_extract = ocr.vision_extract
    original_uploader = st.file_uploader
    ocr.vision_extract = lambda *args, **kwargs: calls.append(args) or (None, "不应被调用")
    st.file_uploader = lambda *args, **kwargs: (
        [FakeUpload(path) for path in images] if kwargs.get("accept_multiple_files") else None
    )
    try:
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240).run()
        sidebar_inputs(at)[app.WIDGET_VISION_MODEL].set_value("deepseek-chat")
        at.run()
        buttons = [button for button in at.button if "用大模型" in str(button.label)]
        check("视觉按钮存在", len(buttons), 1)
        buttons[0].click()
        at.run()
    finally:
        ocr.vision_extract = original_extract
        st.file_uploader = original_uploader
    check("页面无异常", [str(e) for e in at.exception] == [], True)
    warnings = [str(w.value) for w in at.warning]
    expected = "当前视觉模型 deepseek-chat 不支持图片输入，DeepSeek 请用 " + app.DEEPSEEK_VISION_HINT
    check("给出明确提示", any(expected in text for text in warnings), True)
    check("完全没有发起视觉请求", len(calls), 0)
    print("  提示原文:", [text for text in warnings if "deepseek-chat" in text])


if __name__ == "__main__":
    test_offline_no_socket()
    test_parsers()
    test_label_anchoring()
    test_real_images()
    test_app_multi_image()
    test_diagnostics()
    test_vision_payload()
    test_preset_switch()
    test_secrets_config()
    test_secrets_page()
    test_vision_guard_page()
    print("=" * 60)
    if FAILED:
        print("FAILED 项:", FAILED)
        sys.exit(1)
    print("全部检查通过")
