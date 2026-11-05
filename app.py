"""Run Analyzer：跑步数据分析与体能训练方案生成器。

单一分析模式（跑者训练分析）。演示模式直接读取 data/ 下的预置分析结果并渲染，
不调用任何外部 API、不需要任何 Key；它是「没检测到 Key」时的默认值，配好 Key
之后打开页面即走真实 AI（勾选仍可随时切回）。
真实 AI 模式走 OpenAI 兼容 chat.completions，任何异常都会降级为演示结果。

页面顺序：数据接入 → 数据预览 → 趋势图 → 汇总统计 → 跑量统计（周 / 月）
→ 运动评估与训练计划 → 月度复盘（AI）→ 能力档案与后续迭代（折叠）。
"""

import inspect
import json
import re
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import coach_knowledge as coach
import ocr_convert
import record_store
import theme
import training_metrics as metrics

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

MODE_GENERAL = "跑者训练分析"    # 原「通用跑者模式」：现在只保留这一条分析路径

REQUIRED_COLUMNS = ["日期", "里程(km)", "时长(分钟)", "配速(min/km)"]
HEART_RATE_COLUMN = "平均心率"
OPTIONAL_COLUMNS = [HEART_RATE_COLUMN]  # 截图经常没有心率，放宽为可选
NUMERIC_COLUMNS = ["里程(km)", "时长(分钟)", "配速(min/km)"]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEMO_ANALYSIS_FILE = DATA_DIR / "demo_analysis_general.json"
FALLBACK_ANALYSIS = "预置样例结果读取失败：请确认 data/demo_analysis_general.json 存在且为合法 JSON。"
RECORDS_DIR = BASE_DIR / "records"
RECORD_FILE = Path(os.environ.get("RA_RECORD_FILE") or (RECORDS_DIR / "history.csv"))

AI_TIMEOUT_SECONDS = 90.0  # 云端冷启动 + 大记录压缩摘要后首字延迟可达 60 秒，给足余量

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
WIDGET_DEMO = "demo_mode_toggle"
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


def get_demo_analysis() -> str:
    """演示结果的唯一入口；读不到也必须返回可渲染文本，绝不抛异常。"""
    try:
        return load_demo_analysis(str(DEMO_ANALYSIS_FILE))
    except Exception as exc:  # noqa: BLE001 - 演示模式是兜底路径，不能崩
        logger.warning("载入预置分析结果失败: error=%s", type(exc).__name__)
        return FALLBACK_ANALYSIS


# 也接受嵌套写法（Streamlit Cloud 上很多人习惯分节）：[ai] / [deepseek] 下的同名字段
SECRET_SECTIONS = ("ai", "deepseek", "openai")
SECTION_FIELDS = {
    "base_url": "AI_BASE_URL",
    "model": "AI_MODEL",
    "model_name": "AI_MODEL",
    "vision_model": "VISION_MODEL",
    "api_key": "AI_API_KEY",
    "key": "AI_API_KEY",
}


def load_secret_store() -> dict:
    """一次性读取 st.secrets 里与 AI 相关的项；没有 secrets.toml 时返回空 dict。

    同时支持两种写法，谁先出现用谁（顶层优先，避免歧义）：
        AI_API_KEY = "sk-..."                  # 顶层
        [ai]                                   # 或分节
        api_key = "sk-..."
    """
    store = {}
    try:
        for name in SECRET_NAMES:
            value = str(st.secrets.get(name, "") or "").strip()
            if value:
                store[name] = value
        for section in SECRET_SECTIONS:
            block = st.secrets.get(section)
            if not hasattr(block, "get"):
                continue
            for field, target in SECTION_FIELDS.items():
                value = str(block.get(field, "") or "").strip()
                if value and target not in store:
                    store[target] = value
    except Exception as exc:  # noqa: BLE001 - 没有 secrets.toml 时访问 st.secrets 会抛错
        logger.info("未读取到 st.secrets: error=%s", type(exc).__name__)
        return {}
    return store


def provider_label(base_url: str) -> str:
    """把 base_url 归到一个展示用服务商名（只用于界面与日志，不参与鉴权判断）。"""
    lowered = (base_url or "").lower()
    if "deepseek" in lowered:
        return "DeepSeek"
    if "openai" in lowered:
        return "OpenAI"
    return PRESET_CUSTOM


def preset_for_base_url(base_url: str) -> str:
    """把 base_url 归到预设下拉的某一项。

    注意不能拿 provider_label() 的返回值直接当下拉值：它返回 "DeepSeek"，
    而预设项叫 "DeepSeek（推荐，便宜）"——不匹配就会在边栏显示「自定义」，
    与下方「已从 Secrets 检测到 DeepSeek」自相矛盾。
    """
    label = provider_label(base_url)
    if label == "DeepSeek":
        return PRESET_DEEPSEEK
    if label == "OpenAI":
        return PRESET_OPENAI
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


format_seconds = metrics.format_seconds   # 秒 -> 「6分12秒」，与能力档案共用同一套写法


def preview_frame(df: pd.DataFrame) -> pd.DataFrame:
    """数据预览用：日期只显示到「月-日」，其余列保持原样。"""
    view = df.copy()
    if "日期" in view.columns:
        parsed = pd.to_datetime(view["日期"], errors="coerce")
        view["日期"] = parsed.dt.strftime("%m-%d").fillna(view["日期"].astype(str))
    return view


def render_stat_table(items: list) -> None:
    """汇总统计：指标 / 数值两列表格，数值右对齐并等宽，读法与上方数据预览一致。"""
    rows = "".join(
        "<tr><th scope='row'>{0}</th><td>{1}</td></tr>".format(label, value)
        for label, value in items
    )
    st.markdown(
        '<table class="ra-stats"><thead><tr><th>指标</th><th>数值</th></tr></thead>'
        "<tbody>{0}</tbody></table>".format(rows),
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


MARKER_MAX_POINTS = 40    # 超过这个点数不再画记号点，否则糊成一片；同时也是叠滑动平均的门槛


def plot_trend(df: pd.DataFrame, column: str, accent: str, height: int = 300) -> go.Figure:
    """单条趋势线：少量数据是平滑曲线 + 记号点，数据多时再叠一条滑动平均。

    平滑只为读得清：原始记录仍以细线留在图上，真实波动不会被改掉。
    """
    count = len(df)
    figure = px.line(df, x="日期", y=column, markers=count <= MARKER_MAX_POINTS)
    theme.style_figure(figure, accent=accent, height=height)
    if count > MARKER_MAX_POINTS:
        window = max(3, min(14, count // 20))
        average = df[column].rolling(window=window, min_periods=1).mean()
        figure.update_traces(
            line=dict(color=accent, width=0.8, shape="spline", smoothing=0.35),
            marker=dict(size=3, color=accent, opacity=.25),
            opacity=.28,
            name="单次记录",
            showlegend=False,
        )
        figure.add_scatter(
            x=df["日期"], y=average, mode="lines",
            name="{0} 点滑动平均".format(window),
            line=dict(color=accent, width=2.4, shape="spline", smoothing=0.85),
            hovertemplate="%{x|%m-%d}　%{y:.2f}<extra>滑动平均</extra>",
        )
    return figure


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
            st.markdown(
                '<div class="ra-fig-label">{0}</div>'.format(title),
                unsafe_allow_html=True,
            )
            render_figure(plot_trend(df, column, accent))
    if not has_heart_rate(df):
        st.caption("该数据没有心率列，心率趋势图已跳过。")


def render_stats(df: pd.DataFrame) -> None:
    """四格总览：次数 / 里程 / 配速 / 心率，读法与上方数据预览一致。"""
    heart_rate_text = "{0:.0f}".format(df[HEART_RATE_COLUMN].mean()) if has_heart_rate(df) else "—"
    render_stat_table([
        ("跑步次数", "{0}".format(len(df))),
        ("累计里程 km", "{0:.2f}".format(df["里程(km)"].sum())),
        ("平均配速 min/km", "{0:.2f}".format(df["配速(min/km)"].mean())),
        ("平均心率 bpm", heart_rate_text),
    ])


def build_stats_text(df: pd.DataFrame) -> str:
    """送进模型的统计摘要：只放真实算出来的数字，不做推断。"""
    lines = [
        "总跑步次数：{0}".format(len(df)),
        "累计里程：{0:.2f} km".format(df["里程(km)"].sum()),
        "平均配速：{0:.2f} min/km".format(df["配速(min/km)"].mean()),
        "平均心率：{0}".format(
            "{0:.0f} bpm".format(df[HEART_RATE_COLUMN].mean())
            if has_heart_rate(df) else "—（本次数据未提供心率）"),
    ]
    weekly, monthly = metrics.volume_breakdown(df)
    if not weekly.empty:
        lines.append("训练周数：{0} 周，周均 {1:.1f} km，最大单周 {2:.1f} km".format(
            len(weekly), weekly["里程"].mean(), weekly["里程"].max()))
        lines.append("最近 4 周跑量：{0:.1f} km".format(weekly.tail(4)["里程"].sum()))
    if not monthly.empty:
        lines.append("按月跑量：" + "；".join(
            "{0} {1:.1f} km".format(row["月份"].strftime("%Y-%m"), row["里程"])
            for _, row in monthly.iterrows()))
    return "\n".join(lines)


# ---------------------------------------------------------------- 跑量统计

def _volume_frame(table: pd.DataFrame, key_column: str, date_format: str) -> pd.DataFrame:
    """把周/月汇总表整理成给人看的四列。"""
    view = table.copy()
    view[key_column] = pd.to_datetime(view[key_column]).dt.strftime(date_format)
    view = view.rename(columns={key_column: "起始日" if key_column == "周起始" else "月份",
                                "里程": "里程 km", "时长": "时长 分钟"})
    for column, digits in (("里程 km", 2), ("时长 分钟", 0), ("平均配速", 2)):
        view[column] = view[column].round(digits)
    return view


def render_volume_section(df: pd.DataFrame) -> None:
    """周跑量统计 + 月度分布：关键指标用表格，分布用柱状图，明细给可展开的表。"""
    st.subheader("跑量统计 · 周 / 月")
    weekly, monthly = metrics.volume_breakdown(df)
    if weekly.empty:
        st.caption("没有可统计的记录。")
        return

    render_stat_table(metrics.volume_summary(df))

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown('<div class="ra-fig-label">月度跑量分布 (km)</div>', unsafe_allow_html=True)
        bar = go.Figure(
            go.Bar(
                x=monthly["月份"].dt.strftime("%Y-%m"),
                y=monthly["里程"].round(2),
                marker=dict(color=theme.STEEL, line=dict(width=0)),
                text=monthly["里程"].round(1),
                textposition="outside",
                textfont=dict(color=theme.TEXT_MUTED, size=11),
                hovertemplate="%{x}　%{y:.2f} km<extra></extra>",
            )
        )
        theme.apply_base(bar, height=300)
        bar.update_layout(bargap=0.5, showlegend=False)
        bar.update_yaxes(rangemode="tozero")
        # "2026-10" 会被 plotly 当成日期串解析成时间轴（刻度退化成 23:59:59.9996），必须显式声明分类轴
        bar.update_xaxes(type="category")
        render_figure(bar)
    with right:
        st.markdown('<div class="ra-fig-label">每周跑量 (km)</div>', unsafe_allow_html=True)
        line = go.Figure(
            go.Scatter(
                x=weekly["周起始"],
                y=weekly["里程"].round(2),
                mode="lines+markers",
                line=dict(color=theme.SILVER, width=2.2, shape="spline", smoothing=0.6),
                marker=dict(size=5, color=theme.SILVER, line=dict(width=1, color=theme.BG)),
                fill="tozeroy", fillcolor="rgba(211,215,221,.09)",
                hovertemplate="%{x} 周　%{y:.2f} km<extra></extra>",
            )
        )
        theme.apply_base(line, height=300)
        line.update_xaxes(tickformat="%m-%d")
        render_figure(line)

    with st.expander("周 / 月明细表", expanded=False):
        st.caption("周起始为周一；月表按自然月切分。")
        st.dataframe(_volume_frame(weekly, "周起始", "%m-%d"), hide_index=True)
        st.dataframe(_volume_frame(monthly, "月份", "%Y-%m"), hide_index=True)


# ---------------------------------------------------------------- 能力档案

def render_ability_section(df: pd.DataFrame) -> None:
    """六维能力档案：雷达图 + 逐项依据。数据不足的维度显式写「数据不足」而不是补零。"""
    profile = metrics.build_ability_profile(df)
    missing = [item["name"] for item in profile if item["score"] is None]
    # 数据不足的维度不画进多边形：补 0 会在图上造出一个假的短板顶点
    available = [item for item in profile if item["score"] is not None]
    names = [item["name"] for item in available]
    scores = [item["score"] for item in available]

    radar = go.Figure(
        go.Scatterpolar(
            r=scores + scores[:1] if scores else [],
            theta=names + names[:1] if names else [],
            fill="toself",
            fillcolor="rgba(108,129,153,.20)",
            line=dict(color=theme.SILVER, width=2.0),
            marker=dict(size=5, color=theme.SILVER),
            hovertemplate="%{theta}　%{r:.0f}<extra></extra>",
        )
    )
    # 50 分基准环：收缩后的分数大多在 40-70 之间，没有基准线就看不出「这一项是强还是弱」
    radar.add_trace(
        go.Scatterpolar(
            r=[50] * (len(names) + 1) if names else [],
            theta=names + names[:1] if names else [],
            mode="lines",
            line=dict(color="rgba(211,215,221,.28)", width=1.0, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    theme.apply_base(radar, height=380)
    radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 100], showline=False, gridcolor="rgba(255,255,255,.09)",
                            tickfont=dict(color=theme.TEXT_MUTED, size=10), tickvals=[25, 50, 75, 100]),
            angularaxis=dict(gridcolor="rgba(255,255,255,.07)",
                             tickfont=dict(color=theme.TEXT, size=11)),
        ),
        showlegend=False,
        # 六边形顶点上的标签比圆环更靠近边框，左右留 66 才不会被裁掉
        margin=dict(l=66, r=66, t=34, b=30),
    )

    left, right = st.columns([1, 1.05])
    with left:
        if len(available) >= 3:
            render_figure(radar)
        else:
            st.caption("至少三个维度有数据才画雷达图；当前只有 {0} 项可评分。".format(len(available)))
    with right:
        rows = "".join(
            "<tr><th scope='row'>{0}<span class='ra-caliber'>{1}</span></th><td>{2}</td></tr>".format(
                item["name"],
                item["caliber"],
                "—" if item["score"] is None else "{0:.0f}".format(item["score"]),
            )
            for item in profile
        )
        st.markdown(
            '<table class="ra-stats"><thead><tr><th>维度 / 口径</th><th>评分</th></tr></thead>'
            "<tbody>{0}</tbody></table>".format(rows),
            unsafe_allow_html=True,
        )
        for item in profile:
            if item["score"] is None:
                continue
            st.caption("**{0}** · {1}".format(item["name"], item["basis"]))
    if missing:
        st.caption("数据不足：{0} —— 这几项需要更多记录或额外字段，暂不评分。".format("、".join(missing)))
    st.caption(
        "口径说明：评分由本机规则算出并归一化到 0-100，用于找相对短板与观察变化，不是体测结论；"
        "虚线多边形是 50 分基准，落在里面的维度就是当前短板。样本少于 30 次时分数会向 50 收缩，"
        "所以记录越少越不该拿绝对分和别人比。「估算」来自等效 5 公里成绩与 Daniels 公式，"
        "「代理」表示用可得字段近似。"
    )


# ---------------------------------------------------------------- 月度复盘

def render_month_review(df: pd.DataFrame, demo_mode: bool, demo_content: str, settings: dict) -> None:
    """按月复盘：先给本机算出的真实数字，再（可选）让模型补一段文字复盘。"""
    st.subheader("月度复盘")
    months = metrics.month_options(df)
    if not months:
        st.caption("没有可复盘的月份。")
        return

    label = st.selectbox("选择月份", months, index=len(months) - 1, key="month_review_pick")
    month_df = metrics.slice_month(df, label)
    st.dataframe(preview_frame(month_df), hide_index=True)

    stats_text = metrics.month_stats_text(df, label)
    with st.expander("本月统计（本机计算，无需 Key）", expanded=True):
        st.markdown(
            "<div class='ra-pre'>{0}</div>".format(stats_text.replace("\n", "<br>")),
            unsafe_allow_html=True,
        )

    key_name = "month_review_text_" + label
    if demo_mode or not settings["api_key"]:
        st.markdown(demo_content)
        return

    if st.button("用 AI 复盘这个月", key="month_review_run"):
        messages = coach.build_month_messages(label, stats_text, metrics.month_records_csv(df, label))
        try:
            with st.spinner("AI 复盘中，最长等待 {0:.0f} 秒...".format(AI_TIMEOUT_SECONDS)):
                st.session_state[key_name] = call_ai(
                    settings["base_url"], settings["model_name"], settings["api_key"], messages, MODE_GENERAL)
        except Exception as exc:  # noqa: BLE001 - 任何异常都不许打断页面
            logger.warning("月度复盘失败: error=%s detail=%s", type(exc).__name__, exc)
            st.error("AI 复盘失败（{0}）：{1}".format(type(exc).__name__, describe_error(exc)))
            return

    cached = st.session_state.get(key_name)
    if cached:
        st.markdown(cached)
    else:
        st.caption("点击按钮，让模型基于上面的真实数字写一段本月复盘（会明确区分「记录支持的结论」与「数据不足」）。")


# ---------------------------------------------------------------- 迭代预览

def render_roadmap_section(df: pd.DataFrame) -> None:
    """后续迭代预览：能算的先用真实数据画出来，算不了的写清缺哪个字段。"""
    with st.expander("后续迭代预览（负荷 / 睡眠 / 表现趋势）", expanded=False):
        st.caption("这一区是下一版要展开的方向：有真实数据的先画出来，缺字段的明确标出需要什么。")

        load = metrics.build_load_series(df)
        if load is None or load.empty:
            st.caption("30 天训练负荷：记录不足，暂无法计算。")
        else:
            latest = load.iloc[-1]
            ratio = latest["ratio"]
            st.markdown(
                '<div class="ra-fig-label">30 天训练负荷（短期 7 天 / 长期 28 天折周均）</div>',
                unsafe_allow_html=True,
            )
            figure = go.Figure()
            figure.add_scatter(
                x=load["date"], y=load["long"].round(2), mode="lines", name="长期负荷（28 天折周均）",
                line=dict(color=theme.LOAD_LONG, width=2.2, shape="spline", smoothing=0.7),
                hovertemplate="%{x|%m-%d}　%{y:.2f} km<extra>长期</extra>",
            )
            figure.add_scatter(
                x=load["date"], y=load["short"].round(2), mode="lines", name="短期负荷（近 7 天）",
                line=dict(color=theme.LOAD_SHORT, width=2.2, shape="spline", smoothing=0.7),
                hovertemplate="%{x|%m-%d}　%{y:.2f} km<extra>短期</extra>",
            )
            theme.apply_base(figure, height=320)
            figure.update_layout(legend=dict(orientation="h", y=1.12, x=0))
            render_figure(figure)
            st.caption(
                "最近一天：短期 {0:.1f} km，长期 {1:.1f} km，负荷比 {2}。"
                "负荷比长期高于 1.5 通常意味着近期加量偏快，下一版会给出区间提示。".format(
                    latest["short"], latest["long"],
                    "—" if pd.isna(ratio) else "{0:.2f}".format(ratio))
            )

        trend = metrics.performance_trend(df)
        if trend.empty:
            st.caption("表现趋势：需要至少一次 3 km 以上的记录才能估算。")
        else:
            st.markdown(
                '<div class="ra-fig-label">训练表现趋势（估算 VO2max / 跑步表现指数）</div>',
                unsafe_allow_html=True,
            )
            perf = go.Figure()
            perf.add_scatter(
                x=trend["date"], y=trend["vo2"].round(1), mode="lines", name="估算 VO2max (ml/kg/min)",
                line=dict(color=theme.SILVER, width=2.2, shape="spline", smoothing=0.7),
                hovertemplate="%{x|%m-%d}　%{y:.1f}<extra>VO2max</extra>",
            )
            perf.add_scatter(
                x=trend["date"], y=trend["rpi"].round(1), mode="lines", name="跑步表现指数（首次 = 100）",
                line=dict(color=theme.LOAD_SHORT, width=2.0, shape="spline", smoothing=0.7, dash="dot"),
                yaxis="y2",
                hovertemplate="%{x|%m-%d}　%{y:.1f}<extra>RPI</extra>",
            )
            theme.apply_base(perf, height=320)
            perf.update_layout(
                legend=dict(orientation="h", y=1.12, x=0),
                yaxis2=dict(overlaying="y", side="right", showgrid=False,
                            tickfont=dict(color=theme.TEXT_MUTED, size=10), title=None),
            )
            render_figure(perf)
            st.caption("两条线都是估算：由等效 5 公里成绩经 Daniels 公式换算，只用于看自己的变化方向。")

        st.markdown('<div class="ra-fig-label">睡眠质量 × 训练表现（待接入睡眠字段）</div>',
                    unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            st.markdown("**睡眠记录（待接入）**")
            st.dataframe(pd.DataFrame({
                "日期": ["—"], "睡眠时长": ["—"], "入睡时间": ["—"], "睡眠评分": ["—"],
            }), hide_index=True)
        with right:
            st.markdown("**同日训练表现**")
            st.dataframe(pd.DataFrame({
                "日期": ["—"], "平均配速": ["—"], "平均心率": ["—"], "主观 RPE": ["—"],
            }), hide_index=True)
        st.caption(
            "这一块目前没有数据来源：手表的睡眠数据不在截图里，也不在 CSV 四列之内。"
            "下一版计划加「睡眠 CSV 导入」与按日期对齐，再判断睡眠与次日表现的关系。"
        )


AI_DIGEST_MAX_CHARS = 4000   # 送进模型的数据段落硬上限，超出即视为上下文风险
AI_RAW_ROW_LIMIT = 60        # 不超过这个条数的数据直接给原始 CSV，最精确
BUCKET_ROW_LIMIT = 26        # 汇总表最多保留多少行（周汇总或月汇总）
RECENT_ROW_LIMIT = 10        # 摘要里保留最近多少次原始记录


def _rate_text(value) -> str:
    return "{0:.0f}".format(value) if pd.notna(value) else "—"


def build_ai_digest(df: pd.DataFrame) -> str:
    """把任意规模的记录压成有长度上界的摘要：总量 + 趋势对比 + 分段汇总 + 最近若干次。

    送进模型的长度只跟分段数有关，5000 行的 CSV 也不会再撑爆上下文或让模型数不清。
    """
    frame = df.copy()
    frame["_day"] = pd.to_datetime(frame["日期"], errors="coerce").dt.normalize()
    week = frame["_day"] - pd.to_timedelta(frame["_day"].dt.dayofweek, unit="D")
    weekly = week.nunique() <= BUCKET_ROW_LIMIT
    frame["_bucket"] = week if weekly else frame["_day"].values.astype("datetime64[M]")
    has_hr = has_heart_rate(frame)

    lines = [
        "记录条数：{0}（{1} 至 {2}）".format(
            len(frame), frame["_day"].min().strftime("%Y-%m-%d"), frame["_day"].max().strftime("%Y-%m-%d")),
        "累计里程：{0:.2f} km；单次范围 {1:.2f}–{2:.2f} km".format(
            frame["里程(km)"].sum(), frame["里程(km)"].min(), frame["里程(km)"].max()),
        "平均配速：{0:.2f} min/km（最快 {1:.2f}，最慢 {2:.2f}）".format(
            frame["配速(min/km)"].mean(), frame["配速(min/km)"].min(), frame["配速(min/km)"].max()),
        "平均心率：{0}".format(
            "{0:.0f} bpm（{1:.0f}–{2:.0f}）".format(
                frame[HEART_RATE_COLUMN].mean(), frame[HEART_RATE_COLUMN].min(), frame[HEART_RATE_COLUMN].max())
            if has_hr else "—（本次数据未提供心率）"),
    ]
    profile = metrics.build_ability_profile(df)
    lines.append("能力档案（规则评分 0-100，用于找相对短板，不是体检结论）：")
    for item in profile:
        score = "—" if item["score"] is None else "{0:.0f}".format(item["score"])
        lines.append("  {0} {1}｜{2}｜{3}".format(item["name"], score, item["basis"], item["caliber"]))

    buckets = []
    for key, chunk in frame.groupby("_bucket", sort=True):
        heart = chunk[HEART_RATE_COLUMN].mean() if has_hr else float("nan")
        buckets.append((key, len(chunk), chunk["里程(km)"].sum(), chunk["配速(min/km)"].mean(), heart))

    if len(buckets) >= 2:
        recent = buckets[-4:]
        earlier = buckets[-8:-4] or buckets[:4]
        recent_km = sum(item[2] for item in recent) / len(recent)
        earlier_km = sum(item[2] for item in earlier) / len(earlier)
        change = (recent_km / earlier_km - 1) * 100 if earlier_km else 0.0
        lines.append("趋势对比：最近 {0} 期平均 {1:.1f} km，此前 {2} 期平均 {3:.1f} km，变化 {4:+.0f}%".format(
            len(recent), recent_km, len(earlier), earlier_km, change))

    lines.append("")
    lines.append(("{0}：起始日 | 次数 | 里程km | 平均配速 | 平均心率").format("周汇总（每周一行）" if weekly else "月汇总（每月一行）"))
    for key, count, km, pace, heart in buckets[-BUCKET_ROW_LIMIT:]:
        lines.append("{0} | {1} | {2:.1f} | {3:.2f} | {4}".format(
            key.strftime("%Y-%m-%d"), count, km, pace, _rate_text(heart)))

    lines.append("")
    lines.append("最近 {0} 次记录（日期 | 里程km | 配速 | 心率）：".format(min(RECENT_ROW_LIMIT, len(frame))))
    for _, row in frame.tail(RECENT_ROW_LIMIT).iterrows():
        heart = row[HEART_RATE_COLUMN] if has_hr else float("nan")
        lines.append("{0} | {1:.2f} | {2:.2f} | {3}".format(
            row["_day"].strftime("%Y-%m-%d"), row["里程(km)"], row["配速(min/km)"], _rate_text(heart)))

    if has_hr:
        zones = [
            ("低于 130", frame[HEART_RATE_COLUMN] < 130),
            ("130–149", (frame[HEART_RATE_COLUMN] >= 130) & (frame[HEART_RATE_COLUMN] < 150)),
            ("150–169", (frame[HEART_RATE_COLUMN] >= 150) & (frame[HEART_RATE_COLUMN] < 170)),
            ("170 以上", frame[HEART_RATE_COLUMN] >= 170),
        ]
        lines.append("心率分布：" + "　".join("{0} {1} 次".format(label, int(mask.sum())) for label, mask in zones))

    digest = "\n".join(lines)
    if len(digest) > AI_DIGEST_MAX_CHARS:
        digest = digest[:AI_DIGEST_MAX_CHARS] + "\n…（摘要已截断）"
    return digest


def build_ai_payload(df: pd.DataFrame) -> tuple:
    """小数据给原始 CSV（最精确），大数据给压缩摘要（长度有上界）。返回 (文本, 是否压缩)。"""
    raw = df.to_csv(index=False)
    if len(df) <= AI_RAW_ROW_LIMIT and len(raw) <= AI_DIGEST_MAX_CHARS:
        return raw, False
    return build_ai_digest(df), True


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


def ai_status_line(detected: dict, demo_mode: bool, has_manual_key: bool = False) -> str:
    """侧边栏一行状态：现在到底走哪条路、Key 从哪来。"""
    if demo_mode:
        return "当前：演示模式，读预置结果，不联网、不消耗额度。"
    if has_manual_key:
        return "当前：真实 AI，Key 来自上方输入框（仅本次会话有效）。"
    if detected["has_secret_key"]:
        return "当前：真实 AI，Key 来自 Secrets（{0}），重启也不会丢。".format(detected["key_source"])
    return "当前：真实 AI，但还没拿到 Key —— 请在上方填写。"


def render_sidebar() -> tuple:
    with st.sidebar:
        st.header("分析模式")
        st.caption("跑者训练分析：一份记录 → 趋势、跑量、能力档案与训练方案。")
        mode = MODE_GENERAL

        st.markdown("---")
        st.header("AI 设置")
        detected = load_ai_config()

        # 演示模式的初值跟着「有没有可用 Key」走，而不是写死 True：
        # 配好 Secrets 之后打开页面就直接走真实 AI，不用每次手动取消勾选；
        # 没有 Key 的环境仍然默认演示，保证零配置也能看。勾选始终可以手动改回来。
        st.session_state.setdefault(WIDGET_DEMO, not detected["has_secret_key"])
        demo_mode = st.checkbox(
            "演示模式（不调用任何外部 API）",
            key=WIDGET_DEMO,
            help="未检测到 API Key 时默认开启；配置好 Key 后默认关闭，该复选框只是开关，不会清空你的 Key。",
        )

        preset_options = list(PROVIDER_PRESETS) + [PRESET_CUSTOM]
        detected_preset = preset_for_base_url(detected["base_url"]) if detected["has_secret_key"] else PRESET_CUSTOM
        preset = st.selectbox(
            "服务商预设",
            preset_options,
            index=preset_options.index(detected_preset),
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
        st.caption(
            ai_status_line(detected, demo_mode, bool(api_key_input.strip()))
            + " 也可写进 .streamlit/secrets.toml —— 内容不显示、不写日志。"
        )

        st.markdown("---")
        st.header("空间背景")
        space_on = st.checkbox("启用 3D 空间背景（three.js）", value=True)
        space_interactive = st.checkbox(
            "探索模式：拖动旋转视角",
            value=False,
            help="开启后这一层接管鼠标，主内容暂时不可点击（侧边栏仍可用，随时关掉即可恢复）。",
        )
        st.caption("粒子场 + shader 渐变，30 帧上限，切到后台自动暂停；不拦截点击、不遮挡内容。")

    api_key = api_key_input or detected["api_key"]
    return mode, demo_mode, {
        "base_url": (base_url or DEFAULT_BASE_URL).strip(),
        "model_name": (model_name or DEFAULT_MODEL).strip(),
        "vision_model": (vision_model or model_name or DEFAULT_MODEL).strip(),
        "api_key": api_key,
        "space_on": space_on,
        "space_interactive": space_interactive,
    }


def fallback_source():
    """没有任何上传时退回仓库内置样例，保证演示模式永远有数据可看。"""
    sample = BASE_DIR / "run_data.csv"
    if sample.exists():
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
            uploaded = st.file_uploader("上传 CSV",
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


RECORD_FRAME = "record_library"
RECORD_BATCH = "record_batch"
RECORD_NOTICE = "record_notice"
RECORD_SCOPE = "record_scope"
RECORD_SCOPE_SEEN = "record_scope_seen"
ACTIVE_SCOPE = "active_scope"
CLEAR_ARMED = "clear_armed"


def record_library() -> pd.DataFrame:
    """记录库的唯一读入口：本次会话首次访问时尝试从 records/history.csv 恢复。"""
    if RECORD_FRAME not in st.session_state:
        st.session_state[RECORD_FRAME] = record_store.load_file(RECORD_FILE)
    frame = st.session_state[RECORD_FRAME]
    return frame if isinstance(frame, pd.DataFrame) else record_store.empty_frame()


def store_library(frame: pd.DataFrame) -> None:
    """写内存 + 尽力落盘（云端临时磁盘写失败也不影响页面）。"""
    st.session_state[RECORD_FRAME] = frame
    record_store.save_file(RECORD_FILE, frame)


def add_screenshot_records(frame: pd.DataFrame, count: int) -> str:
    """把一批识别结果并进记录库（按指纹去重），返回给用户看的一句话。"""
    batch = record_store.new_batch_id()
    label = "{0} · {1} 张截图".format(record_store.now_text(), count)
    incoming = record_store.prepare(frame, batch, label, "截图识别")
    library, added, skipped = record_store.merge(record_library(), incoming)
    if added:
        st.session_state[RECORD_BATCH] = batch
        store_library(library)
        st.session_state.pop(EDITOR_KEY, None)
    message = "已并入记录库：新增 {0} 条".format(added)
    if skipped:
        message += "，跳过 {0} 条重复记录".format(skipped)
    return message


def record_scope_options() -> list:
    return record_store.scope_options(record_library(), st.session_state.get(RECORD_BATCH))


def record_scope_label(scope: str) -> str:
    for value, label in record_scope_options():
        if value == scope:
            return label
    return "全部历史"


def render_record_scope() -> str:
    """右上角的历史记录选择器：全部历史 / 仅本次上传 / 最近 10 条 / 单个批次。"""
    options = record_scope_options()
    ids = [item[0] for item in options]
    labels = dict(options)
    if st.session_state.get(RECORD_SCOPE) not in ids:
        st.session_state.pop(RECORD_SCOPE, None)      # 选项变了（记录被删/批次消失）就回到默认
    return st.selectbox(
        "历史记录范围",
        ids,
        format_func=lambda value: labels.get(value, value),
        key=RECORD_SCOPE,
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
    """主入口：上传截图 -> 离线/视觉识别 -> 并入记录库 -> 人工校正 -> 导出或接入分析。"""
    uploaded = st.file_uploader(
        "选择跑步记录截图（PNG / JPG）",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="screenshot_uploader",
    )
    render_engine_diagnostics()

    if uploaded:
        payloads = [(item.name, item.getvalue()) for item in uploaded]
        signature = "|".join("{0}:{1}".format(name, len(data)) for name, data in payloads)
        if st.session_state.get(SCREENSHOT_SIGNATURE) != signature:
            with st.spinner("正在离线识别截图（本地推理，不联网）..."):
                frame, notes, failures = convert_screenshots(payloads)
            st.session_state[SCREENSHOT_FRAME] = frame
            st.session_state[SCREENSHOT_NOTES] = notes
            st.session_state[SCREENSHOT_FAILURES] = failures
            st.session_state[SCREENSHOT_SIGNATURE] = signature
            st.session_state[RECORD_NOTICE] = add_screenshot_records(frame, len(payloads))

        for failure in st.session_state.get(SCREENSHOT_FAILURES) or []:
            st.warning(failure)
        for note in st.session_state.get(SCREENSHOT_NOTES) or []:
            st.info(note)
        if st.session_state.get(RECORD_NOTICE):
            st.success(st.session_state[RECORD_NOTICE])

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
                    merged = pd.concat(rows, ignore_index=True)[ocr_convert.COLUMNS]
                    st.session_state[SCREENSHOT_FRAME] = merged
                    st.session_state[RECORD_NOTICE] = add_screenshot_records(merged, len(rows))
                    st.success("大模型识别完成，已并入记录库，请在下面的表格里继续核对。")
                for message in messages:
                    st.warning(message)
    elif not record_library().empty:
        st.caption("没有新截图也可以继续看记录库：下面这张表是之前识别并保存下来的记录。")
    else:
        st.caption("支持跑步 App 的「运动记录详情页」截图。识别后请先在表格里核对，再点「用这份数据开始分析」。")

    library = record_library()
    if library.empty:
        return None

    head_left, head_right = st.columns([1, 1.05])
    with head_left:
        st.markdown("**记录库**")
    with head_right:
        scope = render_record_scope()
    view = record_store.subset(library, scope)
    st.caption(
        "已累积 {0} 次跑步 · 当前显示 {1} 条；在表格里改数值或删行都会同步进记录库，"
        "切换范围只是换个看法，不会丢记录。".format(len(library), len(view))
    )

    if st.session_state.get(RECORD_SCOPE_SEEN) != scope:
        st.session_state[RECORD_SCOPE_SEEN] = scope
        st.session_state.pop(EDITOR_KEY, None)      # 换范围后重建表格，丢掉上一个范围的编辑态

    st.warning("OCR 可能出错，请核对后再使用：表格可以直接改，也可以增删行。")
    edited = st.data_editor(
        view[record_store.COLUMNS],
        num_rows="dynamic",
        key=EDITOR_KEY,
        **_EDITOR_KWARGS,
    )

    if isinstance(edited, pd.DataFrame):
        batch = st.session_state.get(RECORD_BATCH) or record_store.new_batch_id()
        updated = record_store.replace_scope(
            library,
            scope,
            edited,
            batch,
            "表格校正 · {0}".format(record_store.now_text()),
            "手工校正",
        )
        if not record_store.same(updated, library):
            store_library(updated)
            library = updated
            view = record_store.subset(library, scope)

    column_download, column_apply, column_clear = st.columns([1.7, 1.35, 0.95])
    with column_clear:
        if st.session_state.get(CLEAR_ARMED):
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button("确认", key="clear_records_yes", type="primary",
                             help="真的删除记录库里的全部历史"):
                    store_library(record_store.empty_frame())
                    for key in (RECORD_SCOPE, RECORD_SCOPE_SEEN, ACTIVE_SCOPE, ACTIVE_FRAME,
                                SCREENSHOT_SIGNATURE, CLEAR_ARMED):
                        st.session_state.pop(key, None)
                    st.session_state.pop(EDITOR_KEY, None)
                    st.rerun()
            with cancel_col:
                if st.button("取消", key="clear_records_no"):
                    st.session_state.pop(CLEAR_ARMED, None)
                    st.rerun()
        elif st.button("清空", key="clear_records", help="删除记录库里的全部历史，回到仓库内置样例"):
            st.session_state[CLEAR_ARMED] = True
            st.rerun()
    with column_download:
        st.download_button(
            "⬇️ 导出当前范围 CSV",
            data=record_store.csv_bytes(view),
            file_name="run_records_history.csv",
            mime="text/csv",
            key="download_records_csv",
            help="UTF-8 BOM 编码，Excel 双击即可打开，不会中文乱码",
        )
    with column_apply:
        if st.button("✅ 用这份数据开始分析", type="primary", key="apply_screenshot_data",
                     help="下方图表与统计随后只使用当前选中的范围"):
            st.session_state[ACTIVE_SCOPE] = scope
            st.session_state[ACTIVE_FRAME] = record_store.to_analysis(view)
            st.success("已按「{0}」开始分析，下方图表与统计已切换。".format(record_scope_label(scope)))

    if st.session_state.get(CLEAR_ARMED):
        st.warning("再点「确认」才会真的清空记录库；点「取消」保留全部记录。")
    elif st.session_state.get(ACTIVE_SCOPE) is None:
        st.caption("提示：下方图表与统计默认使用「全部历史（{0} 条）」；选中某个范围再点按钮，就只看这一段记录。".format(len(library)))
    return view


def render_analysis(mode: str, demo_mode: bool, demo_content: str, df, settings: dict) -> None:
    """演示模式永远先渲染预置结果；真实模式异常时也回到同一条渲染路径。"""
    st.subheader("运动评估与训练计划")

    if demo_mode:
        render_compact_analysis(demo_content)
        logger.info("分析渲染: mode=%s source=预置样例(演示模式)", mode)
        return

    if not settings["api_key"]:
        render_compact_analysis(demo_content)
        return

    if df is None:
        render_compact_analysis(demo_content)
        return

    if not st.button("生成分析与训练计划", type="primary"):
        st.caption(
            "点击按钮调用真实大模型（{0} · {1}），最长等待 {2:.0f} 秒；"
            "任何异常都会自动降级为演示结果。".format(
                provider_label(settings["base_url"]), settings["model_name"], AI_TIMEOUT_SECONDS
            )
        )
        return

    payload, compressed = build_ai_payload(df)
    if compressed:
        st.caption(
            "数据量较大（{0} 行）：已自动压缩为分段汇总 + 最近 {1} 次记录再送模型，"
            "长度有上界，不会因为记录太多而超上下文失败。".format(len(df), RECENT_ROW_LIMIT)
        )


    messages = coach.build_messages(build_stats_text(df), payload)
    try:
        with st.spinner("AI 分析中，最长等待 {0:.0f} 秒...".format(AI_TIMEOUT_SECONDS)):
            text = call_ai(settings["base_url"], settings["model_name"], settings["api_key"], messages, mode)
    except Exception as exc:  # noqa: BLE001 - 真实模式必须降级而不能崩溃
        logger.warning("AI 调用失败，降级为演示结果: mode=%s error=%s detail=%s", mode, type(exc).__name__, exc)
        st.error("AI 调用失败（{0}）：{1}".format(type(exc).__name__, describe_error(exc)))
        st.markdown(demo_content)
        return

    render_compact_analysis(text)
    logger.info("分析渲染: mode=%s source=真实AI", mode)


def main() -> None:
    st.set_page_config(page_title="RUN ANALYZER · 跑步数据分析与体能训练", page_icon="🏃", layout="wide")
    theme.inject_css()
    theme.render_header()

    mode, demo_mode, settings = render_sidebar()
    theme.render_space_background(settings["space_on"], settings["space_interactive"])
    logger.info("页面渲染: mode=%s demo_mode=%s", mode, demo_mode)

    demo_content = get_demo_analysis()

    df = None
    notes, error = [], None

    # 主入口：一个方框内切换「截图识别 / CSV 文件」，引擎自检也收在里面
    csv_source = render_data_entry(settings)


    library = record_library()
    active_scope = st.session_state.get(ACTIVE_SCOPE)
    active_frame = st.session_state.get(ACTIVE_FRAME)

    if csv_source is not None:
        st.markdown("#### 📄 当前分析数据：本次上传的 CSV")
        df, notes, error = load_run_data(csv_source)
    elif active_scope:
        view = record_store.subset(library, active_scope)
        st.markdown("#### ✅ 当前分析数据：记录库 · {0}".format(record_scope_label(active_scope)))
        df, notes, error = clean_run_dataframe(record_store.to_analysis(view))
    elif not library.empty:
        st.markdown("#### ✅ 当前分析数据：记录库全部历史（{0} 条）".format(len(library)))
        df, notes, error = clean_run_dataframe(record_store.to_analysis(library))
    elif active_frame is not None:
        st.markdown("#### ✅ 当前分析数据：截图识别 + 人工校正结果")
        df, notes, error = clean_run_dataframe(active_frame)
    else:
        source = fallback_source()
        if source is not None:
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

        st.subheader("可视化图表")
        render_general_charts(df)

        st.subheader("汇总统计")
        render_stats(df)

        render_volume_section(df)

    render_analysis(mode, demo_mode, demo_content, df, settings)

    if df is not None:
        render_month_review(df, demo_mode, demo_content, settings)
        with st.expander("跑者能力综合档案（六维雷达）", expanded=False):
            render_ability_section(df)
        render_roadmap_section(df)

    st.markdown("---")
    st.caption("数据仅在本机处理；截图识别默认走本地离线引擎，不发送任何网络请求。")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - 最后一道防线：页面不允许出现未捕获异常
        logger.exception("页面渲染出现未捕获异常: error=%s", type(exc).__name__)
        st.error("页面渲染出现异常，已记录日志。请刷新页面或检查 CSV 数据格式。")
