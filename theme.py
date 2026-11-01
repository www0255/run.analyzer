"""中性灰「严肃跑者」视觉层：配色常量、全局 CSS、Plotly 统一风格。

改版要点（2026-11-02）：
- 配色从荧光绿改为**中性灰**：只有灰阶 + 一个冷银主色，图表靠明度区分而非色相
- 模块标题使用**3D 叠影字**（多层 text-shadow + 顶部高光渐变）
- 数字统一等宽（tabular-nums）
- 图表去网格、去图例噪音，交由页面小节标题表达
"""

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
  font-family: %(mono)s;
  font-size: 2.55rem; font-weight: 800; letter-spacing: 2.5px; margin: 8px 0 6px;
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

/* ===== 3D 可点击控件：立体按压感 ===== */
/* 次级按钮：面上凸起 + 底部实体阴影 */
.stButton > button, .stDownloadButton > button {
  background: linear-gradient(180deg, #24262A 0%%, #1A1C1F 100%%);
  border: 1px solid #33363B; border-top-color: #45484E;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.10), 0 2px 0 #0A0B0D, 0 6px 14px rgba(0,0,0,.45);
  transition: transform .06s ease, box-shadow .06s ease, border-color .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-1px);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.14), 0 3px 0 #0A0B0D, 0 9px 18px rgba(0,0,0,.5);
}
.stButton > button:active, .stDownloadButton > button:active {
  transform: translateY(1px);
  box-shadow: inset 0 2px 5px rgba(0,0,0,.6), 0 1px 0 #0A0B0D;
}
/* 主按钮：银色金属面 + 更强的高光与底影 */
.stButton > button[kind="primary"] {
  background: linear-gradient(180deg, #F0F2F5 0%%, #C6CBD2 55%%, #AEB4BC 100%%);
  border: 1px solid #8A9099; border-top-color: #FFFFFF;
  box-shadow: inset 0 1px 0 #FFFFFF, inset 0 -2px 4px rgba(0,0,0,.18),
              0 3px 0 #6E747C, 0 10px 22px rgba(0,0,0,.55);
  text-shadow: 0 1px 0 rgba(255,255,255,.5);
}
.stButton > button[kind="primary"]:active {
  transform: translateY(2px);
  box-shadow: inset 0 2px 6px rgba(0,0,0,.35), 0 1px 0 #6E747C;
}

/* 单选 / 复选：做成可按压的小卡片 */
[data-testid="stRadio"] [role="radiogroup"] label,
[data-testid="stCheckbox"] label {
  background: linear-gradient(180deg, #1D1F23 0%%, #15171A 100%%);
  border: 1px solid #33363B; border-top-color: #43464C;
  border-radius: 10px; padding: 7px 11px; margin: 2px 0;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.08), 0 2px 0 #0A0B0D, 0 5px 12px rgba(0,0,0,.35);
  transition: transform .06s ease, box-shadow .06s ease, border-color .15s ease;
}
[data-testid="stRadio"] [role="radiogroup"] label:hover,
[data-testid="stCheckbox"] label:hover { border-color: #565A61; transform: translateY(-1px); }
[data-testid="stRadio"] [role="radiogroup"] label:active,
[data-testid="stCheckbox"] label:active {
  transform: translateY(1px); box-shadow: inset 0 2px 5px rgba(0,0,0,.55), 0 1px 0 #0A0B0D;
}
/* 选中项：银色内发光，像被按下并点亮的按钮 */
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked),
[data-testid="stCheckbox"] label:has(input:checked) {
  border-color: #9BA1A9;
  background: linear-gradient(180deg, #2C3036 0%%, #202328 100%%);
  box-shadow: inset 0 0 0 1px rgba(211,215,221,.35), inset 0 2px 6px rgba(0,0,0,.5),
              0 1px 0 #0A0B0D;
}

/* 下拉框：闭合成一块立体控件 */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  background: linear-gradient(180deg, #1D1F23 0%%, #15171A 100%%);
  border: 1px solid #33363B; border-top-color: #43464C; border-radius: 10px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.08), 0 2px 0 #0A0B0D, 0 5px 12px rgba(0,0,0,.35);
}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover { border-color: #565A61; }

/* 文本框 / 数字框：凹进去的输入槽 */
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {
  background: #101214 !important; border-radius: 9px !important;
  box-shadow: inset 0 2px 5px rgba(0,0,0,.65), inset 0 -1px 0 rgba(255,255,255,.05);
}

/* 展开条：立体标题条 */
[data-testid="stExpander"] summary {
  background: linear-gradient(180deg, #1E2024 0%%, #17191C 100%%);
  border-radius: 11px; padding: 8px 12px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.08), 0 2px 0 #0A0B0D;
}
[data-testid="stExpander"] summary:hover { box-shadow: inset 0 1px 0 rgba(255,255,255,.14), 0 2px 0 #0A0B0D, 0 6px 14px rgba(0,0,0,.4); }

/* ===== 四方图：一块立体卡片 + 十字分隔 ===== */
.ra-grid {
  display: grid; border: 1px solid #34373C; border-radius: 14px; overflow: hidden;
  background: linear-gradient(180deg, #1C1E22 0%%, #141619 100%%);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.08), 0 10px 26px rgba(0,0,0,.45);
  margin: 6px 0 4px;
}
.ra-cols-2 { grid-template-columns: repeat(2, 1fr); }
.ra-cols-3 { grid-template-columns: repeat(3, 1fr); }
.ra-cell {
  padding: 16px 18px 14px;
  border-right: 1px solid #262A2F; border-bottom: 1px solid #262A2F;
}
.ra-cols-2 .ra-cell:nth-child(2n) { border-right: none; }
.ra-cols-2 .ra-cell:nth-last-child(-n+2) { border-bottom: none; }
.ra-cols-3 .ra-cell:nth-child(3n) { border-right: none; }
.ra-cols-3 .ra-cell:nth-last-child(-n+3) { border-bottom: none; }
.ra-cell-label {
  font-size: .68rem; letter-spacing: 1.2px; text-transform: uppercase;
  color: var(--ra-muted); margin-bottom: 6px;
}
.ra-cell-value {
  font-family: %(mono)s; font-variant-numeric: tabular-nums;
  font-size: 1.72rem; font-weight: 700; letter-spacing: -0.6px; color: #F5F6F8;
  text-shadow: 0 1px 0 #9BA1A8, 0 2px 6px rgba(0,0,0,.6);
}

/* ===== 按钮：加大立体感（兼容 Streamlit 新版 data-testid 结构） ===== */
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button,
[data-testid="stBaseButton-secondary"] {
  background: linear-gradient(180deg, #33363B 0%%, #232629 45%%, #191B1E 100%%) !important;
  border: 1px solid #3E4247 !important; border-top-color: #5A5F66 !important;
  border-radius: 11px !important; color: #EDEFF2 !important; font-weight: 600 !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.16), inset 0 -2px 4px rgba(0,0,0,.45),
              0 3px 0 #0B0C0E, 0 10px 20px rgba(0,0,0,.5) !important;
  transition: transform .07s ease, box-shadow .07s ease, border-color .15s ease;
}
[data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover {
  border-color: #6A7079 !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.22), inset 0 -2px 4px rgba(0,0,0,.4),
              0 4px 0 #0B0C0E, 0 14px 26px rgba(0,0,0,.55) !important;
}
[data-testid="stButton"] button:active, [data-testid="stDownloadButton"] button:active {
  transform: translateY(2px);
  box-shadow: inset 0 3px 8px rgba(0,0,0,.7), 0 1px 0 #0B0C0E !important;
}
[data-testid="stBaseButton-primary"], [data-testid="stButton"] button[kind="primary"] {
  background: linear-gradient(180deg, #F6F7F9 0%%, #CBD0D6 48%%, #A9AFB7 100%%) !important;
  border: 1px solid #878D95 !important; border-top-color: #FFFFFF !important;
  color: #14161A !important;
  box-shadow: inset 0 1px 0 #FFFFFF, inset 0 -3px 5px rgba(0,0,0,.22),
              0 4px 0 #6C727A, 0 14px 26px rgba(0,0,0,.6) !important;
  text-shadow: 0 1px 0 rgba(255,255,255,.55);
}
[data-testid="stBaseButton-primary"]:active,
[data-testid="stButton"] button[kind="primary"]:active {
  transform: translateY(3px);
  box-shadow: inset 0 3px 8px rgba(0,0,0,.4), 0 1px 0 #6C727A !important;
}

/* 滑块：银色立体手柄 */
[data-testid="stSlider"] [role="slider"] {
  background: linear-gradient(180deg, #F2F4F7 0%%, #C0C6CD 100%%);
  border: 1px solid #7E848C;
  box-shadow: inset 0 1px 0 #FFFFFF, 0 2px 5px rgba(0,0,0,.6);
}
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
    st.markdown(
        '<div class="ra-hero">'
        '<div class="ra-kicker">跑步数据分析与体能训练方案生成器</div>'
        '<div class="ra-title">RUN ANALYZER</div>'
        '<div class="ra-sub">%s</div>'
        "</div>" % subtitle,
        unsafe_allow_html=True,
    )


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
