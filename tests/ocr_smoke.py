"""截图转 CSV 的自测脚本（不依赖 Streamlit）。

运行：python tests/ocr_smoke.py
1) 数值宽容解析单测；2) 用真实截图的坐标版式做标签锚定单测（不跑 OCR）；
3) 有 samples/ 真实截图时跑一次离线 OCR，与人工读图的 oracle 逐项比对。
"""

import sys
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

    payloads = [(path.name, path.read_bytes()) for path in sorted(SAMPLE_DIR.glob("*.jpg"))]
    payloads.append(("broken.jpg", b"this-is-not-an-image"))
    frame, notes, failures = app.convert_screenshots(payloads)
    check("合并后行数", len(frame), 4)
    check("失败提示条数", len(failures), 1)
    for failure in failures:
        print("  失败提示:", failure)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    test_offline_no_socket()
    test_parsers()
    test_label_anchoring()
    test_real_images()
    test_app_multi_image()
    print("=" * 60)
    if FAILED:
        print("FAILED 项:", FAILED)
        sys.exit(1)
    print("全部检查通过")
