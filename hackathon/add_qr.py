# -*- coding: utf-8 -*-
"""把线上地址的二维码放进 deck 的每一页，并可重复执行（幂等）。

用法：python hackathon/add_qr.py
前置：hackathon/qr_badge.png 已生成（segno + 白边）。

做的事：
1. 答辩 deck 每页右上角放一个二维码（品牌 logo 位），右下对齐页码列；
2. 最后一页再加一个大一点的二维码，方便评委直接扫；
3. 一页 PPT 右下角放一个二维码。
重复执行只会调整位置与尺寸，不会重复粘贴。
"""
import pathlib
from pptx import Presentation
from pptx.util import Emu

HERE = pathlib.Path(__file__).resolve().parent
QR = HERE / "qr_badge.png"
DECK = HERE / "答辩多页PPT-dsh.pptx"
ONE = HERE / "一页PPT.pptx"


def place(pic, left_r, top_r, size_r, W, H):
    """按相对坐标摆一个正方形图片（size_r 相对幻灯片宽度）。"""
    side = int(W * size_r)
    pic.width = Emu(side)
    pic.height = Emu(side)
    pic.left = Emu(int(W * left_r) - side)
    pic.top = Emu(int(H * top_r))


def stamp_deck():
    prs = Presentation(DECK)
    W, H = prs.slide_width, prs.slide_height
    for slide in prs.slides:
        logo = [s for s in slide.shapes if s.name == "Brand logo"]
        if not logo:
            continue
        place(logo[0], 0.972, 0.014, 0.068, W, H)
    last = prs.slides[-1]
    big = [s for s in last.shapes if s.name == "QR 大图"]
    if not big:
        pic = last.shapes.add_picture(str(QR), 0, 0)
        pic.name = "QR 大图"
        big = [pic]
    place(big[0], 0.5 + 0.0425, 0.755, 0.085, W, H)
    prs.save(DECK)
    return len(prs.slides)


def stamp_one():
    prs = Presentation(ONE)
    W, H = prs.slide_width, prs.slide_height
    slide = prs.slides[0]
    found = [s for s in slide.shapes if s.name == "QR 一页"]
    if not found:
        pic = slide.shapes.add_picture(str(QR), 0, 0)
        pic.name = "QR 一页"
        found = [pic]
    place(found[0], 0.968, 0.845, 0.062, W, H)
    prs.save(ONE)
    return 1


if __name__ == "__main__":
    print("deck slides:", stamp_deck())
    print("one-pager slides:", stamp_one())
