"""Run Analyzer：跑步数据分析与体能训练方案生成器。

双模式（通用跑者模式 / 警校体能考核模式）。默认演示模式：直接读取 data/ 下的
预置分析结果并渲染，不调用任何外部 API，也不需要任何 Key。
真实 AI 模式走 OpenAI 兼容 chat.completions，任何异常都会降级为演示结果。
"""

import inspect
import json
import re
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import coach_knowledge as coach
import ocr_convert
import theme

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("run_analyzer")

_PLOTLY_SUPPORTS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


def render_figure(figure) -> None:
    """兼容 Streamlit 新旧版本：1.49+ 用 width="stretch"，旧版回退 use_container_width。"""
    if _PLOTLY_SUPPORTS_WIDTH:
        st.plotly_chart(figure, width="stretch")
    else:
        st.plotly_chart(figure, use_container_width=True)

MODE_GENERAL = "通用跑者模式"
MODE_POLICE = "警校体能考核模式"
MODE_OPTIONS = [MODE_GENERAL, MODE_POLICE]

REQUIRED_COLUMNS = ["日期", "里程(km)", "时长(分钟)", "配速(min/km)"]
HEART_RATE_COLUMN = "平均心率"
OPTIONAL_COLUMNS = [HEART_RATE_COLUMN]  # 截图经常没有心率，放宽为可选
NUMERIC_COLUMNS = ["里程(km)", "时长(分钟)", "配速(min/km)"]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEMO_FILES = {
    MODE_GENERAL: DATA_DIR / "demo_analysis_general.json",
    MODE_POLICE: DATA_DIR / "demo_analysis_police.json",
}
FALLBACK_ANALYSIS = (
    "预置样例结果读取失败：请确认 data/demo_analysis_general.json "
    "与 data/demo_analysis_police.json 存在且为合法 JSON。"
)

DEFAULT_GOAL_SECONDS = 270
AI_TIMEOUT_SECONDS = 30.0

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
SECRET_NAMES = ("AI_BASE_URL", "AI_MODEL", "VISION_MODEL", "AI_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY")
SECRET_API_KEY_ORDER = ("AI_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY")

PRESET_DEEPSEEK = "DeepSeek（推荐，便宜）"
PRESET_OPENAI = "OpenAI"
PRESET_CUSTOM = "自定义"
WIDGET_BASE_URL = "ai_base_url"
WIDGET_MODEL = "ai_model_name"
WIDGET_VISION_MODEL = "ai_vision_model"
WIDGET_API_KEY = "ai_api_key_input"
WIDGET_PRESET = "ai_provider_preset"
APPLIED_PRESET_KEY = "ai_applied_preset"

# 预设只写这三个输入框的初值；API Key 永远不被预设触碰
PROVIDER_PRESETS = {
    PRESET_DEEPSEEK: {
        WIDGET_BASE_URL: "https://api.deepseek.com",
        WIDGET_MODEL: "deepseek-flash",
        WIDGET_VISION_MODEL: "deepseek-flash",
    },
    PRESET_OPENAI: {
        WIDGET_BASE_URL: "https://api.openai.com/v1",
        WIDGET_MODEL: "gpt-4o-mini",
        WIDGET_VISION_MODEL: "gpt-4o-mini",
    },
}

# 明显不支持图片输入的模型名（子串匹配）；命中就只提示、绝不发请求
UNSUPPORTED_VISION_TOKENS = ("deepseek-chat", "deepseek-reasoner", "gpt-3.5")
DEEPSEEK_VISION_HINT = "deepseek-flash"


@st.cache_data(show_spinner=False)
def load_demo_analysis(path_str: str) -> str:
    """读取预置分析结果（带缓存，演示模式不触达任何外部接口）。"""
    path = Path(path_str)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, str):
        content = payload
    else:
        content = payload.get("content") or payload.get("analysis") or ""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("预置分析结果缺少 content 字段: " + path.name)
    logger.info("载入预置分析结果: file=%s 字数=%d", path.name, len(content))
    return content


def get_demo_analysis(mode: str) -> str:
    """演示结果的唯一入口；读不到也必须返回可渲染文本，绝不抛异常。"""
    try:
        return load_demo_analysis(str(DEMO_FILES[mode]))
    except Exception as exc:  # noqa: BLE001 - 演示模式是兜底路径，不能崩
        logger.warning("载入预置分析结果失败: mode=%s error=%s", mode, type(exc).__name__)
        return FALLBACK_ANALYSIS


def load_secret_store() -> dict:
    """一次性读取 st.secrets 里与 AI 相关的项；没有 secrets.toml 时返回空 dict。"""
    try:
        values = {name: st.secrets.get(name, "") for name in SECRET_NAMES}
    except Exception as exc:  # noqa: BLE001 - 没有 secrets.toml 时访问 st.secrets 会抛错
        logger.info("未读取到 st.secrets: error=%s", type(exc).__name__)
        return {}
    return {name: str(value or "").strip() for name, value in values.items() if str(value or "").strip()}


def provider_label(base_url: str) -> str:
    """把 base_url 归到一个展示用服务商名（只用于界面与日志，不参与鉴权判断）。"""
    lowered = (base_url or "").lower()
    if "deepseek" in lowered:
        return "DeepSeek"
    if "openai" in lowered:
        return "OpenAI"
    return PRESET_CUSTOM


def load_ai_config(store: dict = None) -> dict:
    """按 AI_* > DEEPSEEK_API_KEY > OPENAI_API_KEY 的优先级解析配置。

    Key 只放在返回值里供请求使用：不打印、不写日志、不回显到页面。
    """
    store = load_secret_store() if store is None else store
    api_key, key_source = "", ""
    for name in SECRET_API_KEY_ORDER:
        if store.get(name):
            api_key, key_source = store[name], name
            break
    config = {
        "base_url": store.get("AI_BASE_URL") or DEFAULT_BASE_URL,
        "model_name": store.get("AI_MODEL") or DEFAULT_MODEL,
        "vision_model": store.get("VISION_MODEL") or store.get("AI_MODEL") or DEFAULT_MODEL,
        "api_key": api_key,
        "key_source": key_source,
        "has_secret_key": bool(api_key),
    }
    if api_key:
        logger.info(
            "已从 st.secrets 检测到配置: 服务商=%s 模型=%s 视觉模型=%s key来源=%s",
            provider_label(config["base_url"]), config["model_name"], config["vision_model"], key_source,
        )
    return config


def vision_guard_message(vision_model: str) -> str:
    """视觉模型能力守卫：明显不支持图片的模型直接拦下，不发请求。空串表示放行。"""
    lowered = (vision_model or "").strip().lower()
    if any(token in lowered for token in UNSUPPORTED_VISION_TOKENS):
        return "当前视觉模型 {0} 不支持图片输入，DeepSeek 请用 {1}".format(
            vision_model.strip(), DEEPSEEK_VISION_HINT
        )
    return ""


def ensure_pace_duration(df: pd.DataFrame, notes: list) -> pd.DataFrame:
    """规格：至少要有 里程+配速 或 里程+时长；缺一半时按物理关系互算补出。"""
    has_pace = "配速(min/km)" in df.columns
    has_duration = "时长(分钟)" in df.columns
    distance = pd.to_numeric(df["里程(km)"], errors="coerce")
    if not has_pace and has_duration:
        df["配速(min/km)"] = pd.to_numeric(df["时长(分钟)"], errors="coerce") / distance
        notes.append("配速列缺失，已用 时长÷里程 自动补出。")
    elif has_pace and not has_duration:
        df["时长(分钟)"] = pd.to_numeric(df["配速(min/km)"], errors="coerce") * distance
        notes.append("时长列缺失，已用 配速×里程 自动补出。")
    return df


def clean_run_dataframe(raw: pd.DataFrame) -> tuple:
    """清洗跑步记录（截图入口与 CSV 入口共用）；返回 (df, 提示列表, 错误信息)。"""
    notes = []
    df = raw.copy()
    if "里程(km)" not in df.columns:
        return None, notes, "缺少必需列：里程(km)。当前列：" + "、".join(map(str, df.columns))

    df = ensure_pace_duration(df, notes)
    if "配速(min/km)" not in df.columns and "时长(分钟)" not in df.columns:
        return None, notes, "至少需要「里程+配速」或「里程+时长」的组合，当前两列都缺失。"

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        logger.warning("缺少必需列: %s", missing)
        return None, notes, "缺少必需列：" + "、".join(missing) + "。当前文件列：" + "、".join(map(str, df.columns))

    total_rows = len(df)

    parsed_dates = pd.to_datetime(df["日期"], errors="coerce", format="mixed")
    bad_dates = int(parsed_dates.isna().sum())
    df = df.assign(日期=parsed_dates)
    if bad_dates:
        df = df.dropna(subset=["日期"])
        notes.append("日期列有 {0} 行无法解析，已跳过这 {0} 行。".format(bad_dates))

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if HEART_RATE_COLUMN in df.columns:
        df[HEART_RATE_COLUMN] = pd.to_numeric(df[HEART_RATE_COLUMN], errors="coerce")
    else:
        df[HEART_RATE_COLUMN] = float("nan")
    if not has_heart_rate(df):
        notes.append("没有可用的心率数据（心率列可选），心率图表与统计已跳过，其余功能正常。")

    rows_before = len(df)
    df = df.dropna(subset=NUMERIC_COLUMNS)
    dropped = rows_before - len(df)
    if dropped:
        notes.append("数值列存在空值或非数字，已剔除 {0} 行。".format(dropped))

    if df.empty:
        logger.warning("清洗后无可用数据: 原始行数=%d", total_rows)
        return None, notes, "清洗后没有可用数据（原始 {0} 行），请检查日期与数值列格式。".format(total_rows)

    df = df.sort_values("日期").reset_index(drop=True)
    logger.info("数据清洗完成: 原始=%d 行 有效=%d 行 提示=%d 条", total_rows, len(df), len(notes))
    return df, notes, None


def load_run_data(source) -> tuple:
    """从 CSV 路径/文件对象读取并清洗（原有入口保持不变）。"""
    notes = []
    try:
        df = pd.read_csv(source)
    except Exception as exc:  # noqa: BLE001
        logger.warning("CSV 解析失败: error=%s", type(exc).__name__)
        return None, notes, "CSV 解析失败（{0}）：请确认文件是带表头的 UTF-8 CSV。".format(type(exc).__name__)
    return clean_run_dataframe(df)


def with_police_columns(df: pd.DataFrame) -> pd.DataFrame:
    """配速(min/km) 直接折算为 1000 米用时（秒）：6.2 min/km -> 372 秒。"""
    converted = df.copy()
    converted["1000米用时(秒)"] = converted["配速(min/km)"] * 60.0
    return converted


def format_seconds(seconds: float) -> str:
    total = int(round(float(seconds)))
    return "{0}分{1:02d}秒".format(total // 60, total % 60)


def preview_frame(df: pd.DataFrame) -> pd.DataFrame:
    """数据预览用：日期只显示到「月-日」，其余列保持原样。"""
    view = df.copy()
    if "日期" in view.columns:
        parsed = pd.to_datetime(view["日期"], errors="coerce")
        view["日期"] = parsed.dt.strftime("%m-%d").fillna(view["日期"].astype(str))
    return view


def render_stat_grid(items: list, per_row: int = 2) -> None:
    """四方图：一块立体卡片里按网格排布指标，格间用十字分隔线。"""
    cells = "".join(
        '<div class="ra-cell"><div class="ra-cell-label">{0}</div>'
        '<div class="ra-cell-value">{1}</div></div>'.format(label, value)
        for label, value in items
    )
    st.markdown(
        '<div class="ra-grid ra-cols-{0}">{1}</div>'.format(per_row, cells),
        unsafe_allow_html=True,
    )


def split_analysis_sections(text: str) -> list:
    """把分析正文按小节切开，返回 [(标题, 正文)]。"""
    sections, title, buffer = [], "", []
    for raw in text.splitlines():
        line = raw.strip().lstrip("#").strip()
        if not line:
            continue
        if re.match(r"^([一二三四五六七八九十]+[、.．]|[0-9]+[、.．])", line):
            if title or buffer:
                sections.append((title, " ".join(buffer)))
            title, buffer = line, []
        else:
            buffer.append(line)
    if title or buffer:
        sections.append((title, " ".join(buffer)))
    return sections


def render_compact_analysis(text: str) -> None:
    """每个小节只露约 30 字，全文收进一个可展开方框，避免页面臃肿。"""
    sections = split_analysis_sections(text)
    if not sections:
        st.markdown(text)
        return
    for title, body in sections[:6]:
        summary = body[:30] + ("…" if len(body) > 30 else "")
        st.markdown("**{0}**　{1}".format(title or "要点", summary))
    with st.expander("展开完整评估与训练计划", expanded=False):
        st.markdown(text)


def has_heart_rate(df) -> bool:
    """心率是可选列：没有列、或整列无有效数值时，相关序列整体跳过。"""
    return HEART_RATE_COLUMN in df.columns and bool(df[HEART_RATE_COLUMN].notna().any())


def render_general_charts(df: pd.DataFrame) -> None:
    """三张独立趋势图（里程 / 配速 / 心率），曲线为平滑插值。"""
    specs = [
        ("里程(km)", "里程趋势 (km)", theme.SILVER),
        ("配速(min/km)", "配速趋势 (min/km，越小越快)", theme.SILVER_DIM),
    ]
    if has_heart_rate(df):
        specs.append((HEART_RATE_COLUMN, "心率趋势 (bpm)", theme.STEEL))

    columns = st.columns(len(specs))
    for container, (column, title, accent) in zip(columns, specs):
        with container:
            figure = px.line(df, x="日期", y=column, markers=True, title=title)
            render_figure(theme.style_figure(figure, accent=accent, height=300))
    if not has_heart_rate(df):
        st.caption("该数据没有心率列，心率趋势图已跳过。")


def render_police_chart(df: pd.DataFrame, goal_seconds: int) -> None:
    figure = px.line(
        df,
        x="日期",
        y="1000米用时(秒)",
        markers=True,
        title="1000 米折算用时趋势（数值越小越快）",
    )
    theme.style_figure(figure, accent=theme.SILVER, height=320)
    figure.add_hline(
        y=goal_seconds,
        line_dash="dash",
        line_color=theme.GOAL,
        annotation_text="参考目标线（可自行设置） {0}".format(format_seconds(goal_seconds)),
        annotation_position="top left",
    )
    render_figure(figure)


def render_stats(df: pd.DataFrame, mode: str, goal_seconds: int) -> None:
    heart_rate_text = "{0:.0f}".format(df[HEART_RATE_COLUMN].mean()) if has_heart_rate(df) else "—"
    render_stat_grid([
        ("跑步次数", "{0}".format(len(df))),
        ("累计里程 km", "{0:.2f}".format(df["里程(km)"].sum())),
        ("平均配速 min/km", "{0:.2f}".format(df["配速(min/km)"].mean())),
        ("平均心率 bpm", heart_rate_text),
    ])

    if mode == MODE_POLICE:
        times = df["1000米用时(秒)"]
        best = float(times.min())
        gap = best - goal_seconds
        render_stat_grid([
            ("最好折算 1000 米", format_seconds(best)),
            ("参考目标线（可自行设置）", format_seconds(goal_seconds)),
            ("与参考目标线的差距", "{0} {1}".format(format_seconds(abs(gap)), "落后" if gap > 0 else "领先")),
        ], per_row=3)


def build_stats_text(df: pd.DataFrame, mode: str, goal_seconds: int) -> str:
    lines = [
        "总跑步次数：{0}".format(len(df)),
        "累计里程：{0:.2f} km".format(df["里程(km)"].sum()),
        "平均配速：{0:.2f} min/km".format(df["配速(min/km)"].mean()),
        "平均心率：{0}".format("{0:.0f} bpm".format(df[HEART_RATE_COLUMN].mean()) if has_heart_rate(df) else "—（本次数据未提供心率）"),
    ]
    if mode == MODE_POLICE:
        lines.append("最好折算 1000 米用时：{0}".format(format_seconds(df["1000米用时(秒)"].min())))
        lines.append("用户自设的参考目标线：{0}（不是任何官方或国家标准）".format(format_seconds(goal_seconds)))
    return "\n".join(lines)


def build_prompt(df: pd.DataFrame, mode: str, stats_text: str, goal_seconds: int) -> list:
    """装配专家级提示词：专家身份、训练学规则与输出格式都在 coach_knowledge.py。"""
    return coach.build_messages(mode, stats_text, df.to_csv(index=False), goal_seconds)


def describe_error(exc: Exception) -> str:
    if isinstance(exc, ImportError):
        return "未安装 openai 依赖，请执行 pip install -r requirements.txt"
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if "timeout" in name or "timeout" in message or "timed out" in message:
        return "请求超时（超过 {0:.0f} 秒）".format(AI_TIMEOUT_SECONDS)
    if "401" in message or "unauthorized" in message or "invalid api key" in message or "authentication" in message:
        return "鉴权失败，请检查 API Key 是否正确"
    if "connect" in message or "network" in message or "getaddrinfo" in message or "dns" in message:
        return "网络不可达，请检查 Base URL 与网络连接"
    if "404" in message or "not found" in message:
        return "接口地址或模型名不存在，请检查 Base URL 与模型名称"
    return "服务返回异常，请稍后重试或更换模型"


def call_ai(base_url: str, model_name: str, api_key: str, messages: list, mode: str) -> str:
    """调用 OpenAI 兼容接口；失败时抛出异常，由调用方统一降级。"""
    from openai import OpenAI  # 延迟导入：依赖缺失也不影响页面加载

    started = time.perf_counter()
    logger.info("AI 调用开始: mode=%s base_url=%s model=%s", mode, base_url, model_name)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=AI_TIMEOUT_SECONDS)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        timeout=AI_TIMEOUT_SECONDS,
    )
    elapsed = time.perf_counter() - started
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("模型返回内容为空")
    logger.info("AI 调用成功: mode=%s model=%s 耗时=%.2fs 输出字数=%d", mode, model_name, elapsed, len(text))
    return text


def seed_widget_defaults(detected: dict) -> None:
    """首帧把 secrets 检测到的值填成输入框初值；之后用户键入的值永远优先。"""
    for widget_key, value in (
        (WIDGET_BASE_URL, detected["base_url"]),
        (WIDGET_MODEL, detected["model_name"]),
        (WIDGET_VISION_MODEL, detected["vision_model"]),
    ):
        st.session_state.setdefault(widget_key, value)


def apply_provider_preset(preset: str) -> None:
    """只在真正切换预设的那一次 rerun 改写三个输入框。

    首帧只记录不覆盖：否则会把 secrets 里的自定义模型名冲掉。API Key 不在预设里，切来切去不丢。
    """
    previous = st.session_state.get(APPLIED_PRESET_KEY)
    st.session_state[APPLIED_PRESET_KEY] = preset
    if previous is None or preset == previous or preset not in PROVIDER_PRESETS:
        return
    st.session_state.update(PROVIDER_PRESETS[preset])


def render_sidebar() -> tuple:
    with st.sidebar:
        st.header("分析模式")
        mode = st.radio("选择模式", MODE_OPTIONS, index=0)
        demo_mode = st.checkbox("演示模式（不调用任何外部 API）", value=True)

        st.markdown("---")
        st.header("AI 设置")
        detected = load_ai_config()

        preset_options = list(PROVIDER_PRESETS) + [PRESET_CUSTOM]
        detected_label = provider_label(detected["base_url"]) if detected["has_secret_key"] else PRESET_CUSTOM
        preset = st.selectbox(
            "服务商预设",
            preset_options,
            index=preset_options.index(detected_label if detected_label in preset_options else PRESET_CUSTOM),
            key=WIDGET_PRESET,
        )
        apply_provider_preset(preset)
        seed_widget_defaults(detected)

        base_url = st.text_input("Base URL", key=WIDGET_BASE_URL)
        model_name = st.text_input("模型名称（文本分析）", key=WIDGET_MODEL)
        vision_model = st.text_input("视觉模型（截图识别）", key=WIDGET_VISION_MODEL)
        if detected["has_secret_key"]:
            st.caption(
                "已从 Secrets 检测到配置（服务商: {0} / 模型: {1}），真实模式默认使用它；下方手动填写仍然优先。".format(
                    provider_label(detected["base_url"]), detected["model_name"]
                )
            )
        api_key_input = st.text_input(
            "API Key", type="password", key=WIDGET_API_KEY, placeholder="仅真实 AI 模式需要"
        )
        st.caption("演示模式无需填写 API Key；也可写进 .streamlit/secrets.toml（内容不会显示、不会写日志）。")

    api_key = api_key_input or detected["api_key"]
    return mode, demo_mode, {
        "base_url": (base_url or DEFAULT_BASE_URL).strip(),
        "model_name": (model_name or DEFAULT_MODEL).strip(),
        "vision_model": (vision_model or model_name or DEFAULT_MODEL).strip(),
        "api_key": api_key,
    }


def fallback_source():
    """没有任何上传时退回仓库内置样例，保证演示模式永远有数据可看。"""
    sample = BASE_DIR / "run_data.csv"
    if sample.exists():
        st.caption("未上传文件，已自动载入仓库内置样例 run_data.csv")
        return str(sample)
    return None


def render_data_entry(settings: dict):
    """数据入口：同一个方框里切换「截图识别 / CSV 文件」，引擎自检也收在这里。"""
    csv_source = None
    with st.container(border=True):
        st.subheader("上传跑步记录")
        entry_mode = st.radio(
            "数据来源（可切换）",
            ["截图识别", "CSV 文件"],
            horizontal=True,
            key="entry_mode",
        )
        if entry_mode == "CSV 文件":
            uploaded = st.file_uploader("上传 CSV（列格式见 README 第 7 节）",
                                        type=["csv"], key="csv_uploader")
            render_engine_diagnostics()
            if uploaded is not None:
                logger.info("数据来源: 上传文件 file=%s", uploaded.name)
                csv_source = uploaded
        else:
            render_screenshot_entry(settings)
    return csv_source


SCREENSHOT_FRAME = "screenshot_frame"
SCREENSHOT_SIGNATURE = "screenshot_signature"
SCREENSHOT_NOTES = "screenshot_notes"
SCREENSHOT_FAILURES = "screenshot_failures"
ACTIVE_FRAME = "active_run_frame"
EDITOR_KEY = "screenshot_editor"
_EDITOR_KWARGS = (
    {"width": "stretch"} if "width" in inspect.signature(st.data_editor).parameters else {"use_container_width": True}
)


def convert_screenshots(payloads: list) -> tuple:
    """逐张离线识别；单张失败不拖垮整批。返回 (DataFrame, 提示列表, 失败说明列表)。"""
    records, notes, failures = [], [], []
    for index, (name, data) in enumerate(payloads, start=1):
        result = ocr_convert.extract_records(data, name)
        if result.records:
            records.extend(result.records)
            notes.extend("第 {0} 张：{1}".format(index, note) for note in result.notes)
            notes.extend(ocr_convert.collect_notes(result.records))
        else:
            records.append({column: None for column in ocr_convert.COLUMNS})
            failures.append("第 {0} 张（{1}）未能识别：{2} 该行已留空，请手动补。".format(index, name, result.error))
        logger.info(
            "截图识别: 序号=%d 文件=%s 引擎=%s 记录=%d 耗时=%.2fs",
            index, name, result.engine, len(result.records), result.elapsed,
        )
    return ocr_convert.records_to_frame(records), notes, failures


@st.cache_data(ttl=600, show_spinner=False)
def get_engine_diagnostics() -> dict:
    """环境自检结果缓存 10 分钟：单纯 rerun（拖滑块等）不必重建引擎。"""
    return ocr_convert.engine_diagnostics()


def render_engine_diagnostics() -> None:
    """可折叠的环境自检：让线上把「OCR 不可用」的真实原因自己说出来（不联网、绝不抛异常）。"""
    with st.expander("环境自检（识别引擎状态）", expanded=False):
        try:
            diagnostics = get_engine_diagnostics()
        except Exception as exc:  # noqa: BLE001 - 自检自身失败也不许影响主流程
            st.caption("自检执行失败：{0}: {1}".format(type(exc).__name__, exc))
            return
        st.dataframe(pd.DataFrame(diagnostics["rows"]), hide_index=True)
        st.markdown("**结论：{0}**".format(diagnostics["verdict"]))


def render_screenshot_entry(settings: dict):
    """主入口：上传截图 -> 离线/视觉识别 -> 人工校正 -> 导出 CSV 或接入分析。"""
    st.caption("截图模式：一张图 = 一次跑步，可一次选多张")
    uploaded = st.file_uploader(
        "选择跑步记录截图（PNG / JPG）",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="screenshot_uploader",
    )
    render_engine_diagnostics()
    if not uploaded:
        st.caption("支持跑步 App 的「运动记录详情页」截图。识别后请先在表格里核对，再点「用这份数据开始分析」。")
        return None

    payloads = [(item.name, item.getvalue()) for item in uploaded]
    signature = "|".join("{0}:{1}".format(name, len(data)) for name, data in payloads)
    if st.session_state.get(SCREENSHOT_SIGNATURE) != signature:
        with st.spinner("正在离线识别截图（本地推理，不联网）..."):
            frame, notes, failures = convert_screenshots(payloads)
        st.session_state[SCREENSHOT_FRAME] = frame
        st.session_state[SCREENSHOT_NOTES] = notes
        st.session_state[SCREENSHOT_FAILURES] = failures
        st.session_state[SCREENSHOT_SIGNATURE] = signature

    for failure in st.session_state.get(SCREENSHOT_FAILURES) or []:
        st.warning(failure)
    for note in st.session_state.get(SCREENSHOT_NOTES) or []:
        st.info(note)

    vision_model = (settings.get("vision_model") or settings.get("model_name") or "").strip()
    st.caption("当前视觉模型：{0}".format(vision_model or "未设置"))
    if st.button("🤖 用大模型识别（更准，需联网）", key="vision_button"):
        guard_message = vision_guard_message(vision_model)
        if guard_message:
            st.warning(guard_message)      # 模型能力不够就只提示，连请求都不发
        else:
            rows, messages = [], []
            for index, (name, data) in enumerate(payloads, start=1):
                vision_frame, vision_error = ocr_convert.vision_extract(
                    data, settings["base_url"], vision_model, settings["api_key"]
                )
                if vision_frame is None:
                    messages.append("第 {0} 张（{1}）：{2}".format(index, name, vision_error))
                else:
                    rows.append(vision_frame)
            if rows:
                st.session_state[SCREENSHOT_FRAME] = pd.concat(rows, ignore_index=True)[ocr_convert.COLUMNS]
                st.session_state.pop(EDITOR_KEY, None)      # 换引擎后丢弃旧表格编辑态
                st.success("大模型识别完成，请在下面的表格里继续核对。")
            for message in messages:
                st.warning(message)

    st.warning("OCR 可能出错，请核对后再使用：表格可以直接改，也可以增删行。")
    edited = st.data_editor(
        st.session_state.get(SCREENSHOT_FRAME),
        num_rows="dynamic",
        key=EDITOR_KEY,
        **_EDITOR_KWARGS,
    )

    column_download, column_apply = st.columns(2)
    with column_download:
        st.download_button(
            "⬇️ 导出 CSV（UTF-8 BOM，Excel 可直接打开）",
            data=ocr_convert.frame_to_csv_bytes(edited),
            file_name="run_records_from_screenshot.csv",
            mime="text/csv",
            key="download_screenshot_csv",
        )
    with column_apply:
        if st.button("✅ 用这份数据开始分析", type="primary", key="apply_screenshot_data"):
            st.session_state[ACTIVE_FRAME] = edited.copy()
            st.success("已使用这份校正后的数据，下方图表与统计已切换。")

    if st.session_state.get(ACTIVE_FRAME) is None:
        st.caption("提示：点「用这份数据开始分析」之后，下方的图表与统计才会改用这份数据。")
    return edited


def render_analysis(mode: str, demo_mode: bool, demo_content: str, df, settings: dict, goal_seconds: int) -> None:
    """演示模式永远先渲染预置结果；真实模式异常时也回到同一条渲染路径。"""
    st.subheader("运动评估与训练计划")

    if demo_mode:
        st.success("演示模式 · 预置样例结果")
        render_compact_analysis(demo_content)
        logger.info("分析渲染: mode=%s source=预置样例(演示模式)", mode)
        return

    if not settings["api_key"]:
        st.warning("真实 AI 模式需要 API Key，当前未填写，已自动使用演示结果。")
        st.success("演示模式 · 预置样例结果")
        render_compact_analysis(demo_content)
        return

    if df is None:
        st.warning("真实 AI 模式需要跑步记录数据，请先上传 CSV；当前已展示演示结果。")
        render_compact_analysis(demo_content)
        return

    if not st.button("生成分析与训练计划", type="primary"):
        st.caption("点击按钮后调用真实大模型；任何异常都会自动降级为演示结果。")
        return

    messages = build_prompt(df, mode, build_stats_text(df, mode, goal_seconds), goal_seconds)
    try:
        with st.spinner("AI 分析中，最长等待 30 秒..."):
            text = call_ai(settings["base_url"], settings["model_name"], settings["api_key"], messages, mode)
    except Exception as exc:  # noqa: BLE001 - 真实模式必须降级而不能崩溃
        logger.warning("AI 调用失败，降级为演示结果: mode=%s error=%s detail=%s", mode, type(exc).__name__, exc)
        st.error("AI 调用失败（{0}）：{1}。已自动降级为演示结果。".format(type(exc).__name__, describe_error(exc)))
        st.success("演示模式 · 预置样例结果")
        st.markdown(demo_content)
        return

    render_compact_analysis(text)
    logger.info("分析渲染: mode=%s source=真实AI", mode)


def main() -> None:
    st.set_page_config(page_title="RUN ANALYZER · 跑步数据分析与体能训练", page_icon="🏃", layout="wide")
    theme.inject_css()
    theme.render_header()

    mode, demo_mode, settings = render_sidebar()
    logger.info("页面渲染: mode=%s demo_mode=%s", mode, demo_mode)

    demo_content = get_demo_analysis(mode)

    df = None
    goal_seconds = DEFAULT_GOAL_SECONDS
    notes, error = [], None

    # 主入口：一个方框内切换「截图识别 / CSV 文件」，引擎自检也收在里面
    csv_source = render_data_entry(settings)

    if demo_mode:
        st.success("演示模式 · 预置样例结果 —— 无需 API Key、不依赖网络。")
    else:
        st.info("真实 AI 模式：将调用外部大模型接口，失败时自动降级为演示结果。")

    source = csv_source if csv_source is not None else fallback_source()

    active_frame = st.session_state.get(ACTIVE_FRAME)
    if active_frame is not None:
        st.markdown("#### ✅ 当前分析数据：截图识别 + 人工校正结果")
        df, notes, error = clean_run_dataframe(active_frame)
    elif source is not None:
        df, notes, error = load_run_data(source)
    else:
        st.info("上传 run_data.csv 开始分析（仓库内置样例数据不可用）。")

    for note in notes:
        st.warning(note)
    if error:
        st.error(error)
        df = None
    elif df is not None:
        st.subheader("数据预览")
        st.dataframe(preview_frame(df), hide_index=True)

        analysis_df = with_police_columns(df) if mode == MODE_POLICE else df

        st.subheader("可视化图表")
        render_general_charts(analysis_df)
        if mode == MODE_POLICE:
            st.markdown("#### 1000 米折算用时与参考目标线")
            goal_seconds = st.slider(
                "参考目标线（可自行设置）",
                min_value=180,
                max_value=420,
                value=DEFAULT_GOAL_SECONDS,
                step=5,
                help="单位：秒。此线只是你自己设置的对照刻度，不代表任何官方或国家标准。",
            )
            st.caption("当前参考目标线：{0}（可拖动滑块自行调整）".format(format_seconds(goal_seconds)))
            render_police_chart(analysis_df, goal_seconds)

        st.subheader("汇总统计")
        render_stats(analysis_df, mode, goal_seconds)

    render_analysis(mode, demo_mode, demo_content, analysis_df_for_prompt(mode, df), settings, goal_seconds)
    st.markdown("---")
    st.caption("数据仅在本机处理；截图识别默认走本地离线引擎，不发送任何网络请求。")


def analysis_df_for_prompt(mode: str, df):
    if df is None:
        return None
    return with_police_columns(df) if mode == MODE_POLICE else df


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - 最后一道防线：页面不允许出现未捕获异常
        logger.exception("页面渲染出现未捕获异常: error=%s", type(exc).__name__)
        st.error("页面渲染出现异常，已记录日志。请刷新页面或检查 CSV 数据格式。")
