"""暗色「严肃跑者」视觉层：配色常量、全局 CSS、Plotly 统一风格。

设计取向（不是霓虹炫技风）：
- 底色接近夜间沥青路，#0B0F14 / #121A23 两级灰阶，长时间读数不刺眼
- 单一强调色（跑道荧光绿 #39D353），只用于数据与操作，不做装饰
- 数字统一等宽（tabular-nums），便于纵向比较配速与秒数
- 图表去网格化、去图例噪音，只留必要刻度与一条基准线
"""

import streamlit as st
import plotly.graph_objects as go

# ---------- 调色板 ----------
BG = "#0B0F14"          # 页面底色
SURFACE = "#121A23"     # 卡片/侧边栏
SURFACE_ALT = "#17202B" # 悬浮/选中
BORDER = "#1F2A37"      # 分隔线
TEXT = "#E6EDF3"        # 主文字
TEXT_MUTED = "#8B97A6"  # 次要文字
ACCENT = "#39D353"      # 跑道荧光绿
ACCENT_SOFT = "#2E7D46"
FOCUS = "#4CC2FF"       # 次要数据（心率等）
GOAL = "#FFB020"        # 参考目标线
DANGER = "#FF5D5D"

FONT_STACK = ('"Inter", "Segoe UI", "PingFang SC", "Microsoft YaHei", '
              '"Noto Sans SC", system-ui, sans-serif')
MONO_STACK = ('"JetBrains Mono", "SF Mono", "Cascadia Mono", Consolas, '
              '"Microsoft YaHei", monospace')

_CSS = """
<style>
:root {
  --ra-bg: %(bg)s; --ra-surface: %(surface)s; --ra-surface-alt: %(surface_alt)s;
  --ra-border: %(border)s; --ra-text: %(text)s; --ra-muted: %(muted)s;
  --ra-accent: %(accent)s; --ra-focus: %(focus)s; --ra-goal: %(goal)s;
}
.stApp { background: var(--ra-bg); }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }
html, body, [class*="css"] { font-family: %(font)s; }

/* 侧边栏：比正文再深一档，用 1px 竖线切分 */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0D131A 0%%, var(--ra-surface) 100%%);
  border-right: 1px solid var(--ra-border);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }

/* 标题与小标签：字距收紧 + 小标签全大写带字距 */
h1 { letter-spacing: -0.4px !important; font-weight: 700 !important; }
h2 { letter-spacing: -0.2px !important; font-weight: 650 !important; }
h3 { letter-spacing: -0.1px !important; font-weight: 600 !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  font-size: 0.76rem !important; text-transform: uppercase;
  letter-spacing: 1.6px !important; color: var(--ra-text) !important;
  opacity: .72;
  font-weight: 600 !important; margin-bottom: .2rem !important;
}

/* 指标卡：分层灰阶 + 等宽数字 */
[data-testid="stMetric"] {
  background: var(--ra-surface); border: 1px solid var(--ra-border);
  border-radius: 10px; padding: 14px 16px 12px;
}
[data-testid="stMetricLabel"] p {
  font-size: .74rem !important; letter-spacing: .8px; text-transform: uppercase;
  color: var(--ra-muted) !important;
}
[data-testid="stMetricValue"] {
  font-family: %(mono)s; font-variant-numeric: tabular-nums;
  font-size: 1.28rem !important; letter-spacing: -0.3px;
  color: var(--ra-text) !important; overflow: visible !important;
}
[data-testid="stMetricDelta"] { font-variant-numeric: tabular-nums; }

/* 收起默认的彩块与阴影，只保留细描边 */
[data-testid="stExpander"] {
  border: 1px solid var(--ra-border) !important; border-radius: 10px !important;
  background: var(--ra-surface);
}
[data-testid="stExpander"] summary { font-size: .86rem; color: var(--ra-muted); }
[data-testid="stFileUploaderDropzone"] {
  background: var(--ra-surface) !important; border: 1px dashed var(--ra-border) !important;
  border-radius: 10px !important;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid var(--ra-border); border-radius: 10px; overflow: hidden;
}

/* 按钮：主操作=荧光绿实心，次操作=描边 */
.stButton > button, .stDownloadButton > button {
  border-radius: 8px; font-size: .88rem; font-weight: 600;
  border: 1px solid var(--ra-border); background: var(--ra-surface-alt);
  color: var(--ra-text); padding: .45rem 1.1rem;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--ra-accent); color: var(--ra-accent);
}
.stButton > button[kind="primary"] {
  background: var(--ra-accent); border-color: var(--ra-accent); color: #06120A;
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.08); color: #06120A; }

/* 提示条：统一压暗，避免五颜六色 */
[data-testid="stAlert"] { border-radius: 10px; border: 1px solid var(--ra-border); }

/* 顶部品牌区 */
.ra-hero { border-bottom: 1px solid var(--ra-border); padding-bottom: 14px; margin-bottom: 18px; }
.ra-kicker {
  font-family: %(mono)s; font-size: .74rem; letter-spacing: 2.4px;
  text-transform: uppercase; color: var(--ra-accent);
}
.ra-title { font-size: 2.05rem; font-weight: 700; letter-spacing: -0.6px; margin: 6px 0 4px; }
.ra-sub { color: var(--ra-muted); font-size: .92rem; }
.ra-rule { height: 1px; background: var(--ra-border); margin: 18px 0; }
.ra-badge {
  display: inline-block; font-family: %(mono)s; font-size: .72rem;
  letter-spacing: 1px; text-transform: uppercase; color: var(--ra-accent);
  border: 1px solid var(--ra-accent); border-radius: 999px;
  padding: 2px 10px; margin-right: 8px;
}
</style>
""" % {
    "bg": BG, "surface": SURFACE, "surface_alt": SURFACE_ALT, "border": BORDER,
    "text": TEXT, "muted": TEXT_MUTED, "accent": ACCENT, "focus": FOCUS,
    "goal": GOAL, "font": FONT_STACK, "mono": MONO_STACK,
}


def inject_css() -> None:
    """注入全局样式；只影响渲染层，不动任何业务逻辑。"""
    st.markdown(_CSS, unsafe_allow_html=True)


def render_header(subtitle: str = "跑步记录 → 分析结论 → 训练方案 · 通用 / 警校体能双模式") -> None:
    st.markdown(
        '<div class="ra-hero">'
        '<div class="ra-kicker">RUN ANALYZER</div>'
        '<div class="ra-title">跑步数据分析与体能训练方案生成器</div>'
        '<div class="ra-sub">%s</div>'
        "</div>" % subtitle,
        unsafe_allow_html=True,
    )


def _template() -> go.layout.Template:
    axis = dict(
        gridcolor=BORDER, zeroline=False, showline=False,
        tickfont=dict(color=TEXT_MUTED, size=11),
        title=None,  # 轴标题交给小节标题表达，图上不留重复文字
    )
    return go.layout.Template(layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, color=TEXT, size=12),
        colorway=[ACCENT, FOCUS, GOAL, DANGER],
        xaxis=axis, yaxis=axis,
        margin=dict(l=8, r=8, t=34, b=8),
        height=280,
        hovermode="x unified",
        hoverlabel=dict(bgcolor=SURFACE_ALT, bordercolor=BORDER,
                        font=dict(color=TEXT, family=FONT_STACK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=TEXT_MUTED, size=11)),
        title=dict(font=dict(color=TEXT, size=13), x=0, xanchor="left"),
    ))


def style_figure(figure, accent: str = ACCENT, height: int = 280):
    """把 Plotly 图统一到暗色皮肤：细网格、荧光主线、紧凑边距。"""
    figure.update_layout(template=_template(), height=height)
    figure.update_traces(
        line=dict(color=accent, width=2),
        marker=dict(size=5, color=accent, line=dict(width=1, color=BG)),
        hovertemplate="%{x|%m-%d}　%{y}<extra></extra>",
    )
    return figure
