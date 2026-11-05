"""记录库：把每次识别的跑步记录累积起来，按「全部历史 / 最近 N 条 / 单批次」查看。

为什么需要它：一张截图只能读出一次跑步，但训练判断必须看累积记录。所以识别结果
不再只停留在「本次上传」，而是并进一个记录库，表格默认展示全部历史。

落盘位置 records/history.csv（UTF-8 BOM，Excel 可直接打开）。真正的持久层是
session_state —— Streamlit Cloud 的容器磁盘是临时的，重启即清空，文件只是本地
运行的备份与人工导出物：想恢复就把导出的 CSV 放回这个路径，下次本地启动会自动读回。
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
from pathlib import Path

import pandas as pd

COLUMNS = ["日期", "里程(km)", "时长(分钟)", "配速(min/km)", "平均心率"]
META_BATCH = "批次ID"
META_BATCH_LABEL = "批次"
META_ADDED = "录入时间"
META_SOURCE = "来源"
META_COLUMNS = [META_BATCH, META_BATCH_LABEL, META_ADDED, META_SOURCE]
ALL_COLUMNS = COLUMNS + META_COLUMNS

NUMERIC_COLUMNS = ["里程(km)", "时长(分钟)", "配速(min/km)", "平均心率"]

SCOPE_ALL = "__all__"
SCOPE_RECENT = "__recent__"
RECENT_COUNT = 10

BATCH_PREFIX = "batch::"


def now_text() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def new_batch_id() -> str:
    return hashlib.sha1(_dt.datetime.now().isoformat().encode("utf-8")).hexdigest()[:8]


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=ALL_COLUMNS)


def _date_text(value) -> str:
    """日期归一成 YYYY-MM-DD 参与指纹；解析不了就退回原文小写去空格。"""
    stamp = pd.to_datetime(pd.Series([value]), errors="coerce", format="mixed").iloc[0]
    if pd.notna(stamp):
        return stamp.strftime("%Y-%m-%d")
    return str(value).strip().lower()


def _number_text(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number != number:          # NaN
        return ""
    return "{0:.3f}".format(number)


def fingerprint(row) -> str:
    """一次跑步的身份：日期 + 里程 + 时长 + 配速。心率可选，不参与判定。"""
    parts = [
        _date_text(row.get("日期")),
        _number_text(row.get("里程(km)")),
        _number_text(row.get("时长(分钟)")),
        _number_text(row.get("配速(min/km)")),
    ]
    return "|".join(parts)


def _has_any(row) -> bool:
    return any(str(row.get(column) or "").strip() for column in COLUMNS[:2])


def prepare(frame: pd.DataFrame, batch_id: str, batch_label: str, source: str,
            added_at: str | None = None) -> pd.DataFrame:
    """把编辑器/CSV 里的原始行整理成记录库行（补批次元数据、丢掉全空行）。"""
    if frame is None or len(frame) == 0:
        return empty_frame()
    working = frame.copy()
    for column in COLUMNS:
        if column not in working.columns:
            working[column] = None
    working = working[COLUMNS]
    keep = working.apply(_has_any, axis=1)
    working = working[keep.fillna(False)].reset_index(drop=True)
    if working.empty:
        return empty_frame()
    working[META_BATCH] = batch_id
    working[META_BATCH_LABEL] = batch_label
    working[META_ADDED] = added_at or now_text()
    working[META_SOURCE] = source
    return working[ALL_COLUMNS]


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    """记录库的规范顺序：按日期升序，同一天按录入时间。表格、图表、导出都吃这个顺序。

    pandas 的 sort_values 是稳定排序，所以同日期同录入时间的行保留原有先后。
    """
    if frame is None or frame.empty:
        return empty_frame()
    view = frame.reindex(columns=ALL_COLUMNS).copy()
    view["_ord"] = pd.to_datetime(view["日期"], errors="coerce")
    view["_added"] = view[META_ADDED].astype(str)
    view = view.sort_values(["_ord", "_added"], kind="stable", na_position="last")
    return view.drop(columns=["_ord", "_added"]).reset_index(drop=True)


def merge(existing: pd.DataFrame, incoming: pd.DataFrame) -> tuple:
    """并入记录库，按指纹去重（保留先入库的那条）。返回 (合并结果, 新增数, 跳过数)。"""
    if incoming is None or incoming.empty:
        base = existing if existing is not None else empty_frame()
        return ordered(base), 0, 0
    base = existing if existing is not None and not existing.empty else empty_frame()
    for column in ALL_COLUMNS:
        if column not in base.columns:
            base[column] = None
    combined = pd.concat([base[ALL_COLUMNS], incoming[ALL_COLUMNS]], ignore_index=True)
    tokens = combined.apply(fingerprint, axis=1)
    deduped = combined.assign(_fp=tokens).drop_duplicates(subset="_fp", keep="first")
    deduped = deduped.drop(columns=["_fp"]).reset_index(drop=True)
    added = len(deduped) - len(base)
    skipped = len(incoming) - added
    return ordered(deduped), max(added, 0), max(skipped, 0)


def batches(frame: pd.DataFrame) -> list:
    """[{'id','label','count','added_at'}]，按录入时间倒序。"""
    if frame is None or frame.empty:
        return []
    rows = []
    for batch_id, chunk in frame.groupby(META_BATCH, sort=False):
        rows.append({
            "id": str(batch_id),
            "label": str(chunk[META_BATCH_LABEL].iloc[0]),
            "count": len(chunk),
            "added_at": str(chunk[META_ADDED].iloc[0]),
        })
    rows.sort(key=lambda item: item["added_at"], reverse=True)
    return rows


def scope_options(frame: pd.DataFrame, local_batch: str | None = None) -> list:
    """下拉框选项 [(scope_id, 显示文案)]，第一项永远是「全部历史」。"""
    total = 0 if frame is None or frame.empty else len(frame)
    options = [(SCOPE_ALL, "全部历史（{0} 条）".format(total))]
    local_count = 0
    if local_batch:
        found = [item for item in batches(frame) if item["id"] == local_batch]
        local_count = found[0]["count"] if found else 0
    if local_count:
        options.append((BATCH_PREFIX + str(local_batch), "仅本次上传（{0} 条）".format(local_count)))
    if total > RECENT_COUNT:
        options.append((SCOPE_RECENT, "最近 {0} 条".format(RECENT_COUNT)))
    for batch in batches(frame):
        if local_batch and batch["id"] == local_batch:
            continue                      # 已经以「仅本次上传」出现过，不再重复
        options.append((BATCH_PREFIX + batch["id"],
                        "{0} · {1} 条".format(batch["label"], batch["count"])))
    return options


def subset(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    """按选择范围切出记录；返回仍带元数据列的视图。"""
    if frame is None or frame.empty:
        return empty_frame()
    if scope == SCOPE_RECENT:
        latest = ordered(frame).tail(RECENT_COUNT)
        return latest.reset_index(drop=True)
    if scope and scope.startswith(BATCH_PREFIX):
        batch_id = scope[len(BATCH_PREFIX):]
        return ordered(frame[frame[META_BATCH].astype(str) == batch_id])
    return ordered(frame)


def drop_scope(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    """subset 的反操作：把当前范围整块摘掉，剩下的是要原样保留的记录。"""
    if frame is None or frame.empty:
        return empty_frame()
    if scope == SCOPE_RECENT:
        recent = frame.sort_values("日期", ascending=False).head(RECENT_COUNT)
        tokens = set(fingerprint(row) for _, row in recent.iterrows())
        stay = frame.apply(fingerprint, axis=1).isin(tokens) == False   # noqa: E712
        return frame[stay].reset_index(drop=True)
    if scope and scope.startswith(BATCH_PREFIX):
        batch_id = scope[len(BATCH_PREFIX):]
        return frame[frame[META_BATCH].astype(str) != batch_id].reset_index(drop=True)
    return empty_frame()


def replace_scope(frame: pd.DataFrame, scope: str, edited: pd.DataFrame,
                  batch_id: str, batch_label: str, source: str = "手工校正") -> pd.DataFrame:
    """把「当前范围」替换成编辑后的内容，其他范围的记录一字不动。

    指纹还对得上的行保留原来的批次信息（只改数值不会被标成新批次）；
    新出现的行才记到 batch_id 名下。
    """
    scope_rows = subset(frame, scope)
    keep = drop_scope(frame, scope)
    original = {}
    if scope_rows is not None and not scope_rows.empty:
        for _, row in scope_rows.iterrows():
            original[fingerprint(row)] = row
    prepared = prepare(edited, batch_id, batch_label, source)
    for index, row in prepared.iterrows():
        meta = original.get(fingerprint(row))
        if meta is not None:
            for column in META_COLUMNS:
                prepared.at[index, column] = meta[column]
    merged, _added, _skipped = merge(keep, prepared)
    return merged


def same(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """两份记录库内容是否等价（逐格比字符串，忽略索引与 dtype 抖动）。"""
    def table(frame):
        if frame is None or frame.empty:
            return []
        view = frame.reindex(columns=ALL_COLUMNS)
        return [tuple(str(value if value is not None else "") for value in row)
                for row in view.itertuples(index=False)]
    return table(left) == table(right)


def to_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    """剥掉元数据列，交给清洗函数；数值列统一转成数值类型。"""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=COLUMNS)
    working = frame[COLUMNS].copy().reset_index(drop=True)
    working["日期"] = working["日期"].astype(str)
    for column in NUMERIC_COLUMNS:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    return working


def csv_bytes(frame: pd.DataFrame) -> bytes:
    view = frame if frame is not None else empty_frame()
    return view.to_csv(index=False).encode("utf-8-sig")


def from_csv(data) -> pd.DataFrame:
    """读回导出的 CSV；缺列补齐，多余列忽略。"""
    try:
        frame = pd.read_csv(data)
    except Exception:  # noqa: BLE001 - 导入失败不该影响页面
        return empty_frame()
    if frame.empty:
        return empty_frame()
    for column in ALL_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame[META_BATCH] = frame[META_BATCH].fillna("imported")
    frame[META_BATCH_LABEL] = frame[META_BATCH_LABEL].fillna("导入的历史记录")
    frame[META_ADDED] = frame[META_ADDED].fillna(now_text())
    frame[META_SOURCE] = frame[META_SOURCE].fillna("导入")
    return ordered(frame[ALL_COLUMNS])


def load_file(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return empty_frame()
    try:
        return from_csv(str(path))
    except Exception:  # noqa: BLE001
        return empty_frame()


def save_file(path: Path, frame: pd.DataFrame) -> bool:
    """尽力落盘：云端只读/临时目录失败也只返回 False，不影响页面。"""
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".tmp")
        temp.write_bytes(csv_bytes(frame))
        os.replace(temp, target)
        return True
    except Exception:  # noqa: BLE001
        return False
