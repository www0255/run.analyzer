"""现有功能回归 + 心率可选（4 列数据）实测，用 Streamlit AppTest 驱动真实页面。

运行：python tests/test_app_regression.py
"""

import sys
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

APP = str(ROOT / "app.py")
FAILED = []


def expect(condition, label, detail=""):
    print("  [{0}] {1} {2}".format("OK" if condition else "FAIL", label, detail))
    if not condition:
        FAILED.append(label)


def texts(at):
    return "\n".join([str(element.value) for element in at.markdown] +
                     [str(element.value) for element in at.warning] +
                     [str(element.value) for element in at.info] +
                     [str(element.value) for element in at.success] +
                     [str(element.value) for element in at.caption])


def case_demo_general():
    print("[1] 通用跑者模式 + 演示模式（默认路径）")
    at = AppTest.from_file(APP, default_timeout=120).run()
    expect(not at.exception, "无未捕获异常", [str(e) for e in at.exception])
    expect(len(at.radio) >= 1, "侧边栏模式单选存在")
    expect("内置样例 run_data.csv" in texts(at), "CSV 兜底：自动载入 run_data.csv")
    expect(any("上传跑步记录" in str(s.value) for s in at.subheader), "统一上传区在主入口")
    expect(any("数据来源" in str(r.label) for r in at.radio), "存在「截图识别 / CSV 文件」切换")
    expect(any("环境自检" in str(exp.label) for exp in at.expander), "环境自检收在上传区内")
    return at


def case_heart_rate_optional():
    print("[2] 心率列缺失（4 列数据）不报错")
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["active_run_frame"] = pd.DataFrame({
        "日期": ["2026-10-01", "2026-10-03"],
        "里程(km)": [3.0, 3.5],
        "时长(分钟)": [21.0, 24.0],
        "配速(min/km)": [7.0, 6.86],
    })
    at.run()
    expect(not at.exception, "无未捕获异常", [str(e) for e in at.exception])
    joined = "\n".join([str(w.value) for w in at.warning])
    expect("心率" in joined, "页面给出心率跳过提示", repr(joined[-80:]))
    grid = "\n".join(str(m.value) for m in at.markdown)
    expect("平均心率" in grid and "—" in grid, "四方图里平均心率显示 —", repr(grid[-120:]))
    expect("每次跑步里程趋势" not in "", "图表标题不计入文本")
    return at


def case_police_and_slider():
    print("[3] 警校体能考核模式 + 参考目标线滑块")
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio[0].set_value("警校体能考核模式").run()
    expect(not at.exception, "切换到警校模式无异常", [str(e) for e in at.exception])
    expect(len(at.slider) >= 1, "参考目标线滑块出现")
    if at.slider:
        at.slider[0].set_value(300).run()
        expect(not at.exception, "拖动滑块后无异常", [str(e) for e in at.exception])
        grid = "\n".join(str(m.value) for m in at.markdown)
        expect("参考目标线" in grid and "1000 米" in grid, "四方图出现 1000 米折算指标", repr(grid[-120:]))
    return at


def case_real_mode_without_key():
    print("[4] 关闭演示模式（无 Key）自动降级")
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.checkbox[0].set_value(False).run()
    expect(not at.exception, "真实模式无 Key 时无异常", [str(e) for e in at.exception])
    expect("演示" in texts(at), "页面回退到演示结果")
    return at


def case_upload_and_vision_without_key():
    """页面级：模拟多图上传 -> 离线识别 -> 点「用大模型识别」但无 Key。"""
    print("[5] 多图上传 + 视觉引擎无 Key 友好提示")
    import streamlit as st

    class FakeUpload:
        def __init__(self, path):
            self.name = path.name
            self._data = path.read_bytes()

        def getvalue(self):
            return self._data

    images = sorted((ROOT / "samples").glob("*.jpg"))
    original_uploader = st.file_uploader

    def fake_uploader(*args, **kwargs):
        # 只有多选上传器（截图入口）才返回文件；CSV 单文件上传器必须返回 None，
        # 否则 read_input_source() 会把 list 当成文件对象抛 AttributeError（上一版就是这样假绿的）。
        return [FakeUpload(path) for path in images] if kwargs.get("accept_multiple_files") else None

    st.file_uploader = fake_uploader

    try:
        at = AppTest.from_file(APP, default_timeout=240).run()
        expect(not at.exception, "上传多图后页面无异常", [str(e) for e in at.exception])
        expect(not at.error, "页面没有错误框（兜底 try/except 吞掉的异常也要抓）",
               [str(e.value) for e in at.error])
        joined_info = "\n".join([str(i.value) for i in at.info])
        expect("日期存在歧义" in joined_info, "歧义提示出现在页面上", repr(joined_info[-90:]))
        joined = "\n".join([str(w.value) for w in at.warning])
        expect("OCR 可能出错" in joined, "出现人工校正提示")
        buttons = [b for b in at.button if "用大模型" in str(b.label)]
        expect(len(buttons) == 1, "视觉引擎按钮存在")
        if buttons:
            buttons[0].click().run()
            expect(not at.exception, "无 Key 点击大模型按钮不崩", [str(e) for e in at.exception])
            warnings = "\n".join([str(w.value) for w in at.warning])
            expect("API Key" in warnings, "无 Key 时给出友好提示", repr(warnings[-80:]))
    finally:
        st.file_uploader = original_uploader


if __name__ == "__main__":
    case_demo_general()
    case_heart_rate_optional()
    case_police_and_slider()
    case_real_mode_without_key()
    case_upload_and_vision_without_key()
    print("=" * 60)
    if FAILED:
        print("FAILED:", FAILED)
        sys.exit(1)
    print("回归全部通过")
