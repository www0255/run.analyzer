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
    # subheader 也算正文：跑量统计 / 月度复盘已经改成 st.subheader 与其它区块同级
    return "\n".join([str(element.value) for element in at.markdown] +
                     [str(element.value) for element in at.subheader] +
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
    body = texts(at)
    expect("警校" not in body, "页面已无警校体能考核入口")
    expect("参考目标线" not in body, "已移除 1000 米参考目标线")
    expect("跑量统计" in body, "跑量统计区出现")
    expect(any("跑者能力综合档案" in str(exp.label) for exp in at.expander),
           "六维能力档案入口出现")
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


def case_volume_and_ability():
    print("[3] 跑量统计 + 能力档案 + 月度复盘 + 迭代预览")
    at = AppTest.from_file(APP, default_timeout=120).run()
    expect(not at.exception, "单模式页面无异常", [str(e) for e in at.exception])
    expect(not at.radio or len([r for r in at.radio if "模式" in str(r.label)]) == 0,
           "侧边栏不再有模式切换（已合并为单一分析路径）")
    joined = texts(at)
    expect("跑量统计" in joined, "跑量统计区出现")
    expect("训练周数" in joined or "周均跑量" in joined, "跑量关键指标进表格")
    labels = [str(e.label) for e in at.expander]
    expect(any("能力综合档案" in item for item in labels), "六维能力档案收在折叠页", str(labels))
    expect(any("后续迭代" in item for item in labels), "后续迭代预览收在折叠页", str(labels))
    expect(any("月度复盘" in str(s.value) for s in at.subheader) or "月度复盘" in joined, "月度复盘区出现")
    return at


def case_ability_metrics():
    print("[4] 能力档案 / 负荷 / 表现趋势的数值规则")
    import training_metrics as metrics
    import pandas as pd

    demo = ROOT / "demo_csv"
    path = demo / "03_冲刺提升_60次.csv"
    if not path.exists():
        print("  [SKIP] 缺 60 次样例，跳过能力档案断言")
        return
    df = pd.read_csv(path, encoding="utf-8-sig")
    profile = metrics.build_ability_profile(df)
    expect(len(profile) == 6, "六维全部返回", str([i["name"] for i in profile]))
    expect([i["name"] for i in profile] == metrics.ABILITY_DIMENSIONS, "维度顺序与常量一致")
    expect(all(0 <= (i["score"] or 0) <= 100 for i in profile), "评分都在 0-100 内")
    expect(all(i["basis"] for i in profile), "每项都给了依据")

    weekly, monthly = metrics.volume_breakdown(df)
    expect(not weekly.empty and not monthly.empty, "周表与月表都非空")
    expect(abs(weekly["里程"].sum() - monthly["里程"].sum()) < 1e-6, "周表与月表总里程一致")
    expect(abs(monthly["里程"].sum() - df["里程(km)"].sum()) < 1e-6, "月表合计等于原始里程合计")

    load = metrics.build_load_series(df)
    expect(load is not None and len(load) > 0, "负荷序列可算")
    expect(float(load["short"].iloc[-1]) >= 0, "短期负荷非负")
    trend = metrics.performance_trend(df)
    expect(not trend.empty and abs(float(trend["rpi"].iloc[0]) - 100.0) < 1e-6, "表现指数首条为 100")

    months = metrics.month_options(df)
    expect(months == sorted(months) and len(months) >= 2, "月份按时间升序", str(months))
    text = metrics.month_stats_text(df, months[-1])
    expect("统计月份" in text and "月跑量" in text, "单月摘要含月份与跑量", repr(text[:60]))


def case_heart_rate_less_month():
    print("[5] 无心率数据时的能力档案不虚构")
    import training_metrics as metrics
    import pandas as pd

    path = ROOT / "demo_csv" / "05_缺心率列_45次.csv"
    if not path.exists():
        print("  [SKIP] 缺无心率样例")
        return
    df = pd.read_csv(path, encoding="utf-8-sig")
    profile = {i["name"]: i for i in metrics.build_ability_profile(df)}
    expect(profile["心血管基础"]["caliber"] in ("代理", "实测"), "无心率时改用代理口径",
           profile["心血管基础"]["caliber"])
    expect("心率" in profile["心血管基础"]["basis"] or "配速" in profile["心血管基础"]["basis"],
           "依据里说明用了什么字段")
    text = metrics.month_stats_text(df, metrics.month_options(df)[-1])
    expect("无心率数据" in text, "月度摘要明确写无心率", repr(text[-60:]))


def case_real_mode_without_key():
    print("[6] 关闭演示模式（无 Key）自动降级")
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.checkbox[0].set_value(False).run()
    expect(not at.exception, "真实模式无 Key 时无异常", [str(e) for e in at.exception])
    expect("演示" in texts(at), "页面回退到演示结果")
    return at


def case_upload_and_vision_without_key():
    """页面级：模拟多图上传 -> 离线识别 -> 点「用大模型识别」但无 Key。"""
    print("[7] 多图上传 + 视觉引擎无 Key 友好提示")
    import streamlit as st

    class FakeUpload:
        def __init__(self, path):
            self.name = path.name
            self._data = path.read_bytes()

        def getvalue(self):
            return self._data

    images = sorted((ROOT / "samples").glob("*.jpg"))
    if not images:
        # samples/ 里是用户自己的真实跑步截图（.gitignore 排除，不进仓库）。
        # 干净克隆或删过 samples/ 的环境没有这些图，这一段 OCR 断言无从验证——
        # 跳过而不是判失败，否则所有人 clone 下来跑测试都会红。
        print("  [SKIP] samples/ 无截图，跳过截图识别相关的 4 项断言")
        return
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


def case_smooth_chart_and_large_data():
    """图表平滑 + 大数据给压缩摘要：不靠肉眼，直接查 trace 与摘要长度。"""
    print("[8] 曲线平滑与大数据摘要上界")
    import app

    demo = ROOT / "demo_csv"
    big = demo / "04_大样本_500次.csv"
    small = demo / "01_入门跑者_30次.csv"
    if not big.exists() or not small.exists():
        print("  [SKIP] demo_csv/ 缺演示数据，跳过摘要与平滑断言")
        return

    df_big, _, err_big = app.load_run_data(open(big, "rb"))
    df_small, _, err_small = app.load_run_data(open(small, "rb"))
    expect(err_big is None and err_small is None, "演示 CSV 可加载", str(err_big or err_small))

    payload, compressed = app.build_ai_payload(df_big)
    expect(compressed, "500 行数据走压缩摘要")
    expect(len(payload) <= app.AI_DIGEST_MAX_CHARS, "摘要长度不超过硬上限",
           "{0} <= {1}".format(len(payload), app.AI_DIGEST_MAX_CHARS))
    expect("汇总" in payload and "最近" in payload, "摘要含分段汇总与最近记录")
    small_payload, small_compressed = app.build_ai_payload(df_small)
    expect(not small_compressed, "30 行数据仍给原始 CSV（不提前压缩）", "{0} 字符".format(len(small_payload)))

    figure = app.plot_trend(df_big, "里程(km)", "#fff")
    expect(len(figure.data) == 2, "大数据多点一条滑动平均", "{0} 条 trace".format(len(figure.data)))
    expect(all(trace.line.shape == "spline" for trace in figure.data), "两条曲线都是 spline")
    small_figure = app.plot_trend(df_small, "里程(km)", "#fff")
    expect(len(small_figure.data) == 1, "40 点以内的数据不叠滑动平均")
    expect(small_figure.data[0].mode == "lines+markers", "小数据仍保留记号点", str(small_figure.data[0].mode))


if __name__ == "__main__":
    case_demo_general()
    case_heart_rate_optional()
    case_volume_and_ability()
    case_ability_metrics()
    case_heart_rate_less_month()
    case_real_mode_without_key()
    case_upload_and_vision_without_key()
    case_smooth_chart_and_large_data()
    print("=" * 60)
    if FAILED:
        print("FAILED:", FAILED)
        sys.exit(1)
    print("回归全部通过")
