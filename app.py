"""Run Analyzer：跑步数据分析与体能训练方案生成器。

双模式（通用跑者模式 / 警校体能考核模式）。默认演示模式：直接读取 data/ 下的
预置分析结果并渲染，不调用任何外部 API，也不需要任何 Key。
真实 AI 模式走 OpenAI 兼容 chat.completions，任何异常都会降级为演示结果。
"""

import inspect
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import ocr_convert

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


def get_secret_api_key() -> str:
    """st.secrets 里配置的 OPENAI_API_KEY（未配置时返回空串）。"""
    try:
        value = st.secrets.get("OPENAI_API_KEY", "")
    except Exception as exc:  # noqa: BLE001 - 没有 secrets.toml 时访问 st.secrets 会抛错
        logger.info("未读取到 st.secrets: error=%s", type(exc).__name__)
        return ""
    return str(value or "")


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


def has_heart_rate(df) -> bool:
    """心率是可选列：没有列、或整列无有效数值时，相关序列整体跳过。"""
    return HEART_RATE_COLUMN in df.columns and bool(df[HEART_RATE_COLUMN].notna().any())


def render_general_charts(df: pd.DataFrame) -> None:
    specs = [
        ("里程(km)", "每次跑步里程趋势"),
        ("配速(min/km)", "配速变化（数值越小越快）"),
    ]
    if has_heart_rate(df):
        specs.append((HEART_RATE_COLUMN, "平均心率变化"))
    columns = st.columns(len(specs))
    for container, (column, title) in zip(columns, specs):
        with container:
            figure = px.line(df, x="日期", y=column, markers=True, title=title)
            render_figure(figure)
    if not has_heart_rate(df):
        st.caption("该数据没有心率列，心率变化图已跳过。")


def render_police_chart(df: pd.DataFrame, goal_seconds: int) -> None:
    figure = px.line(
        df,
        x="日期",
        y="1000米用时(秒)",
        markers=True,
        title="1000 米折算用时趋势（数值越小越快）",
    )
    figure.add_hline(
        y=goal_seconds,
        line_dash="dash",
        line_color="#d62728",
        annotation_text="参考目标线（可自行设置） {0}".format(format_seconds(goal_seconds)),
        annotation_position="top left",
    )
    render_figure(figure)


def render_stats(df: pd.DataFrame, mode: str, goal_seconds: int) -> None:
    columns = st.columns(4)
    columns[0].metric("跑步次数", "{0} 次".format(len(df)))
    columns[1].metric("累计里程", "{0:.2f} km".format(df["里程(km)"].sum()))
    columns[2].metric("平均配速", "{0:.2f} min/km".format(df["配速(min/km)"].mean()))
    heart_rate_text = "{0:.0f} bpm".format(df[HEART_RATE_COLUMN].mean()) if has_heart_rate(df) else "—"
    columns[3].metric(HEART_RATE_COLUMN, heart_rate_text)

    if mode == MODE_POLICE:
        times = df["1000米用时(秒)"]
        best = float(times.min())
        gap = best - goal_seconds
        police_columns = st.columns(3)
        police_columns[0].metric("最好折算 1000 米", format_seconds(best))
        police_columns[1].metric("参考目标线（可自行设置）", format_seconds(goal_seconds))
        police_columns[2].metric(
            "与参考目标线的差距",
            "{0}（{1}）".format(format_seconds(abs(gap)), "落后" if gap > 0 else "领先"),
        )


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


def build_prompt(df: pd.DataFrame, mode: str, stats_text: str, goal_seconds: int) -> str:
    if mode == MODE_POLICE:
        task = (
            "你是一位警校体能训练的助理教练。请基于上面的记录输出三部分：\n"
            "1. 1000 米折算成绩评估（说明最好、最差与波动情况）；\n"
            "2. 与用户自设参考目标线之间的差距，以及主要短板；\n"
            "3. 下周 7 天训练课表（每周最多两次速度刺激，其余为有氧与恢复）。\n"
            "注意：参考目标线只是用户自己设置的对照刻度，不要把它说成任何官方标准或考核要求。\n"
            "文字简洁，适合直接展示在网页上。"
        )
    else:
        task = (
            "你是一位面向学生跑者的路跑教练。请输出三部分：\n"
            "1. 简要评估近期运动状态；\n"
            "2. 心率风险提示；\n"
            "3. 一份下周简易训练计划，兼顾恢复，强度不要过高。\n"
            "文字简洁，适合直接展示在网页上。"
        )
    return "{0}\n\n原始记录：\n{1}\n\n{2}".format(stats_text, df.to_csv(index=False), task)


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


def call_ai(base_url: str, model_name: str, api_key: str, prompt: str, mode: str) -> str:
    """调用 OpenAI 兼容接口；失败时抛出异常，由调用方统一降级。"""
    from openai import OpenAI  # 延迟导入：依赖缺失也不影响页面加载

    started = time.perf_counter()
    logger.info("AI 调用开始: mode=%s base_url=%s model=%s", mode, base_url, model_name)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=AI_TIMEOUT_SECONDS)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        timeout=AI_TIMEOUT_SECONDS,
    )
    elapsed = time.perf_counter() - started
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("模型返回内容为空")
    logger.info("AI 调用成功: mode=%s model=%s 耗时=%.2fs 输出字数=%d", mode, model_name, elapsed, len(text))
    return text


def render_sidebar() -> tuple:
    with st.sidebar:
        st.header("⚙️ 分析模式")
        mode = st.radio("选择模式", MODE_OPTIONS, index=0)
        demo_mode = st.checkbox("演示模式（不调用任何外部 API）", value=True)

        st.markdown("---")
        st.header("🤖 AI 设置")
        secret_key = get_secret_api_key()
        base_url = st.text_input("Base URL", value="https://api.openai.com/v1")
        model_name = st.text_input("模型名称", value="gpt-4o-mini")
        api_key_input = st.text_input("API Key", type="password", placeholder="仅真实 AI 模式需要")
        if secret_key:
            st.caption("已检测到 st.secrets 中的 OPENAI_API_KEY，真实模式将默认使用它。")
        else:
            st.caption("演示模式无需填写 API Key。")

    api_key = api_key_input or secret_key
    return mode, demo_mode, {"base_url": base_url.strip(), "model_name": model_name.strip(), "api_key": api_key}


def read_input_source():
    uploaded = st.file_uploader("上传跑步记录 CSV", type=["csv"])
    if uploaded is not None:
        logger.info("数据来源: 上传文件 file=%s", uploaded.name)
        return uploaded
    sample = BASE_DIR / "run_data.csv"
    if sample.exists():
        st.caption("未上传文件，已自动载入仓库内置样例 run_data.csv")
        return str(sample)
    return None


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
    with st.expander("🔧 环境自检（识别引擎状态）", expanded=False):
        try:
            diagnostics = get_engine_diagnostics()
        except Exception as exc:  # noqa: BLE001 - 自检自身失败也不许影响主流程
            st.caption("自检执行失败：{0}: {1}".format(type(exc).__name__, exc))
            return
        st.dataframe(pd.DataFrame(diagnostics["rows"]), hide_index=True)
        st.markdown("**结论：{0}**".format(diagnostics["verdict"]))


def render_screenshot_entry(settings: dict):
    """主入口：上传截图 -> 离线/视觉识别 -> 人工校正 -> 导出 CSV 或接入分析。"""
    st.subheader("📷 上传跑步记录截图（PNG/JPG）")
    uploaded = st.file_uploader(
        "可以一次选多张（一张截图 = 一次跑步）",
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

    if st.button("🤖 用大模型识别（更准，需联网）", key="vision_button"):
        rows, messages = [], []
        for index, (name, data) in enumerate(payloads, start=1):
            vision_frame, vision_error = ocr_convert.vision_extract(
                data, settings["base_url"], settings["model_name"], settings["api_key"]
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
    st.subheader("🤖 运动评估与训练计划")

    if demo_mode:
        st.success("演示模式 · 预置样例结果")
        st.markdown(demo_content)
        logger.info("分析渲染: mode=%s source=预置样例(演示模式)", mode)
        return

    if not settings["api_key"]:
        st.warning("真实 AI 模式需要 API Key，当前未填写，已自动使用演示结果。")
        st.success("演示模式 · 预置样例结果")
        st.markdown(demo_content)
        return

    if df is None:
        st.warning("真实 AI 模式需要跑步记录数据，请先上传 CSV；当前已展示演示结果。")
        st.markdown(demo_content)
        return

    if not st.button("生成分析与训练计划", type="primary"):
        st.caption("点击按钮后调用真实大模型；任何异常都会自动降级为演示结果。")
        return

    prompt = build_prompt(df, mode, build_stats_text(df, mode, goal_seconds), goal_seconds)
    try:
        with st.spinner("AI 分析中，最长等待 30 秒..."):
            text = call_ai(settings["base_url"], settings["model_name"], settings["api_key"], prompt, mode)
    except Exception as exc:  # noqa: BLE001 - 真实模式必须降级而不能崩溃
        logger.warning("AI 调用失败，降级为演示结果: mode=%s error=%s detail=%s", mode, type(exc).__name__, exc)
        st.error("AI 调用失败（{0}）：{1}。已自动降级为演示结果。".format(type(exc).__name__, describe_error(exc)))
        st.success("演示模式 · 预置样例结果")
        st.markdown(demo_content)
        return

    st.markdown(text)
    logger.info("分析渲染: mode=%s source=真实AI", mode)


def main() -> None:
    st.set_page_config(page_title="跑步数据分析与体能训练方案生成器", page_icon="🏃", layout="wide")
    st.title("🏃 跑步数据分析与体能训练方案生成器")
    st.caption("Run Analyzer · 通用跑者模式 / 警校体能考核模式 · 支持零配置演示与真实大模型分析")

    mode, demo_mode, settings = render_sidebar()
    logger.info("页面渲染: mode=%s demo_mode=%s", mode, demo_mode)

    demo_content = get_demo_analysis(mode)

    df = None
    goal_seconds = DEFAULT_GOAL_SECONDS
    notes, error = [], None

    # 主入口：页面顶部第一个交互就是截图上传
    render_screenshot_entry(settings)

    if demo_mode:
        st.success("演示模式 · 预置样例结果 —— 无需 API Key、不依赖网络。")
    else:
        st.info("真实 AI 模式：将调用外部大模型接口，失败时自动降级为演示结果。")

    with st.expander("兼容：直接上传 CSV 文件", expanded=False):
        source = read_input_source()

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
        st.subheader("📋 数据预览（已清洗）")
        st.dataframe(df)

        analysis_df = with_police_columns(df) if mode == MODE_POLICE else df

        st.subheader("📈 可视化图表")
        render_general_charts(analysis_df)
        if mode == MODE_POLICE:
            st.markdown("#### 🎯 1000 米折算用时与参考目标线")
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

        st.subheader("📊 汇总统计")
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
