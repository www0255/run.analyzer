"""three.js 空间背景层：粒子场 + shader 渐变，固定在内容之下。

Streamlit 的 markdown 会剥掉 <script>，所以只能走 components.v1.html 的 iframe；
theme.py 的 CSS 把这个 iframe 钉成全屏固定层（z-index:0 / pointer-events:none），
主内容容器被抬到 z-index:2，所以背景永远不挡点击、也不遮挡内容。
"""

from pathlib import Path

import streamlit.components.v1 as components

_TEMPLATE_PATH = Path(__file__).with_name("space_bg.html")
_template = None


def _template_html() -> str:
    global _template
    if _template is None:
        _template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    return _template


def render(interactive: bool = False, height: int = 1) -> None:
    """渲染背景组件。interactive=True 时开启拖动探索（这一层会接管指针事件）。"""
    flag = "true" if interactive else "false"
    components.html(_template_html().replace("__INTERACTIVE__", flag), height=height, scrolling=False)
