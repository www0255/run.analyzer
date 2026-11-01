"""中性灰「严肃跑者」视觉层：配色常量、全局 CSS、Plotly 统一风格。

改版要点（2026-11-02）：
- 配色从荧光绿改为**中性灰**：只有灰阶 + 一个冷银主色，图表靠明度区分而非色相
- 模块标题使用**3D 叠影字**（多层 text-shadow + 顶部高光渐变）
- 数字统一等宽（tabular-nums）
- 图表去网格、去图例噪音，交由页面小节标题表达
"""

from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

# ---------- 中性灰调色板 ----------
BG = "#0D0E10"            # 页面底色（近黑）
SURFACE = "#161719"       # 卡片
SURFACE_ALT = "#1E2023"   # 悬浮/次级
BORDER = "#2C2E33"        # 分隔线
TEXT = "#E9EAEC"          # 主文字
TEXT_MUTED = "#8F9298"    # 次要文字
SILVER = "#D3D7DD"        # 主数据线（冷银）
SILVER_DIM = "#7C8188"    # 次数据线
STEEL = "#6C8199"         # 心率（低饱和钢蓝，唯一有色相的数据）
GOAL = "#A8AEB6"          # 参考目标线（浅灰虚线）
DANGER = "#C4756B"        # 风险提示（低饱和砖红）

ASSETS = Path(__file__).resolve().parent / "assets"
RUNNER_GIF = ASSETS / "runner.gif"

FONT_STACK = ('"Inter", "Segoe UI", "PingFang SC", "Microsoft YaHei", '
              '"Noto Sans SC", system-ui, sans-serif')
MONO_STACK = ('"JetBrains Mono", "SF Mono", "Cascadia Mono", Consolas, '
              '"Microsoft YaHei", monospace')

_CSS = """
<style>
:root {
  --ra-bg: %(bg)s; --ra-surface: %(surface)s; --ra-surface-alt: %(surface_alt)s;
  --ra-border: %(border)s; --ra-text: %(text)s; --ra-muted: %(muted)s;
  --ra-silver: %(silver)s; --ra-steel: %(steel)s;
}
.stApp { background: var(--ra-bg); }
.block-container { padding-top: 2.0rem; padding-bottom: 3rem; max-width: 1180px; }
html, body, [class*="css"] { font-family: %(font)s; }

/* 侧边栏：再深一档 + 1px 竖线 */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #101113 0%%, var(--ra-surface) 100%%);
  border-right: 1px solid var(--ra-border);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.3rem; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  font-size: 0.72rem !important; text-transform: uppercase;
  letter-spacing: 1.8px !important; color: var(--ra-muted) !important;
  font-weight: 600 !important; text-shadow: none !important;
}

/* ===== 3D 叠影标题 ===== */
.ra-hero { border-bottom: 1px solid var(--ra-border); padding-bottom: 16px; margin-bottom: 20px; }
.ra-kicker {
  font-family: %(mono)s; font-size: .72rem; letter-spacing: 3px;
  text-transform: uppercase; color: var(--ra-muted);
}
.ra-title {
  font-size: 2.1rem; font-weight: 800; letter-spacing: -0.8px; margin: 8px 0 6px;
  color: #F2F3F5;
  text-shadow:
    0 1px 0 #C9CDD3, 0 2px 0 #A9AEB5, 0 3px 0 #8B9098, 0 4px 0 #6E737A,
    0 5px 0 #555A61, 0 8px 14px rgba(0,0,0,.65), 0 1px 1px rgba(255,255,255,.18);
}
.ra-sub { color: var(--ra-muted); font-size: .9rem; }

/* 正文模块标题：h2/h3 用弱化版 3D 叠影，避免抢主标题 */
h2, [data-testid="stMarkdownContainer"] h2 {
  font-weight: 750 !important; letter-spacing: -0.3px !important;
  color: #EDEEF0 !important;
  text-shadow: 0 1px 0 #B9BEC5, 0 2px 0 #9AA0A8, 0 3px 6px rgba(0,0,0,.55);
}
h3, [data-testid="stMarkdownContainer"] h3 {
  font-weight: 700 !important; letter-spacing: -0.1px !important;
  color: #E4E6E9 !important;
  text-shadow: 0 1px 0 #A7ADB4, 0 2px 4px rgba(0,0,0,.5);
}
.ra-h3d {
  font-size: 1.32rem; font-weight: 800; letter-spacing: -0.2px; margin: 4px 0 10px;
  color: #EFF1F3;
  text-shadow: 0 1px 0 #B4B9C0, 0 2px 0 #969CA3, 0 3px 8px rgba(0,0,0,.6);
}

/* ===== 指标「四方图」 ===== */
[data-testid="stMetric"] {
  background: linear-gradient(180deg, var(--ra-surface-alt) 0%%, var(--ra-surface) 100%%);
  border: 1px solid var(--ra-border); border-radius: 12px;
  padding: 16px 18px 14px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.04), 0 6px 18px rgba(0,0,0,.35);
}
[data-testid="stMetricLabel"] p {
  font-size: .70rem !important; letter-spacing: 1.1px; text-transform: uppercase;
  color: var(--ra-muted) !important;
}
[data-testid="stMetricValue"] {
  font-family: %(mono)s; font-variant-numeric: tabular-nums;
  font-size: 1.72rem !important; letter-spacing: -0.5px;
  color: #F4F5F7 !important; overflow: visible !important;
  text-shadow: 0 1px 0 #9BA1A8, 0 2px 6px rgba(0,0,0,.6);
}

/* 收起默认彩块与阴影 */
[data-testid="stExpander"] {
  border: 1px solid var(--ra-border) !important; border-radius: 12px !important;
  background: var(--ra-surface);
}
[data-testid="stExpander"] summary { font-size: .84rem; color: var(--ra-muted); }
[data-testid="stFileUploaderDropzone"] {
  background: var(--ra-surface) !important; border: 1px dashed var(--ra-border) !important;
  border-radius: 12px !important;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid var(--ra-border); border-radius: 12px; overflow: hidden;
}

/* 按钮：主操作=浅银实心，次操作=描边 */
.stButton > button, .stDownloadButton > button {
  border-radius: 9px; font-size: .86rem; font-weight: 600;
  border: 1px solid var(--ra-border); background: var(--ra-surface-alt);
  color: var(--ra-text); padding: .45rem 1.1rem;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--ra-silver); color: #FFFFFF;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(180deg, #E6E9ED 0%%, #C3C8CF 100%%);
  border-color: #B7BCC3; color: #14161A;
  box-shadow: 0 2px 0 #9AA0A8, 0 6px 14px rgba(0,0,0,.45);
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.06); color: #14161A; }
[data-testid="stAlert"], [data-testid="stAlertContainer"] {
  border-radius: 12px !important; border: 1px solid var(--ra-border) !important;
  background: var(--ra-surface) !important; color: var(--ra-text) !important;
}
[data-testid="stAlert"] svg, [data-testid="stAlertContainer"] svg { color: var(--ra-muted) !important; }
[data-testid="stImage"] img { border-radius: 12px; }
</style>
""" % {
    "bg": BG, "surface": SURFACE, "surface_alt": SURFACE_ALT, "border": BORDER,
    "text": TEXT, "muted": TEXT_MUTED, "silver": SILVER, "steel": STEEL,
    "font": FONT_STACK, "mono": MONO_STACK,
}


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def section_title(text: str) -> None:
    """正文模块标题（3D 叠影字），替代 st.subheader。"""
    st.markdown('<div class="ra-h3d">%s</div>' % text, unsafe_allow_html=True)


def render_header(subtitle: str = "跑步记录 → 分析结论 → 训练方案 · 通用 / 警校体能双模式") -> None:
    left, right = st.columns([5, 1])
    with left:
        st.markdown(
            '<div class="ra-hero">'
            '<div class="ra-kicker">RUN ANALYZER</div>'
            '<div class="ra-title">跑步数据分析与体能训练方案生成器</div>'
            '<div class="ra-sub">%s</div>'
            "</div>" % subtitle,
            unsafe_allow_html=True,
        )
    with right:
        if RUNNER_GIF.exists():
            st.image(str(RUNNER_GIF), width=180, caption=None)


def _template() -> go.layout.Template:
    axis = dict(
        gridcolor="rgba(255,255,255,0.05)", zeroline=False, showline=False,
        tickfont=dict(color=TEXT_MUTED, size=11), title=None,
    )
    return go.layout.Template(layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, color=TEXT, size=12),
        colorway=[SILVER, STEEL, SILVER_DIM, DANGER],
        xaxis=axis, yaxis=axis,
        margin=dict(l=10, r=12, t=18, b=10),
        height=300,
        hovermode="x unified",
        hoverlabel=dict(bgcolor=SURFACE_ALT, bordercolor=BORDER,
                        font=dict(color=TEXT, family=FONT_STACK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_MUTED, size=11)),
    ))


def apply_base(figure, height: int = 300):
    """只套用模板与尺寸，保留自定义 trace（多图对比时用）。"""
    figure.update_layout(template=_template(), height=height)
    return figure


def style_figure(figure, accent: str = SILVER, height: int = 300, smooth: bool = True):
    """统一图表皮肤：细网格、平滑曲线、紧凑边距、克制记号点。"""
    figure.update_layout(template=_template(), height=height)
    figure.update_traces(
        line=dict(color=accent, width=2.2, shape="spline", smoothing=0.6),
        marker=dict(size=4, color=accent, line=dict(width=1, color=BG), opacity=.75),
        hovertemplate="%{x|%m-%d}　%{y:.2f}<extra></extra>",
    )
    return figure
