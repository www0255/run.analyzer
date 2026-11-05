"""训练指标计算：跑量统计、能力档案、负荷与表现趋势、单月复盘。

纯计算模块，不依赖 Streamlit —— 页面只负责把结果画出来，规则本身可以单独复算和单测。
所有 0-100 分都是可复现的规则评分，用来找相对短板，不是体能测评结论。
"""

import numpy as np
import pandas as pd

HEART_RATE_COLUMN = "平均心率"
DISTANCE = "里程(km)"
DURATION = "时长(分钟)"
PACE = "配速(min/km)"

# 能力档案的六个维度（顺序即雷达图的顶点顺序）
ABILITY_DIMENSIONS = ["心血管基础", "最大摄氧量", "耐力", "速度", "恢复质量", "训练纪律"]


# ---------------------------------------------------------------- 时间骨架

def format_seconds(seconds) -> str:
    total = int(round(float(seconds)))
    return "{0}分{1:02d}秒".format(total // 60, total % 60)


NUMERIC_COLUMNS = [DISTANCE, DURATION, PACE, HEART_RATE_COLUMN]


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """把数值列强制转成数字。脏数据里的 "6分10" / 空串在这里变成 NaN，后面的聚合才不会炸。"""
    frame = df.copy()
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def add_time_keys(df: pd.DataFrame) -> pd.DataFrame:
    """补上 _day / _week / _month 三个时间键，其余列原样保留（数值列先做一次类型清洗）。"""
    frame = coerce_numeric(df)
    day = pd.to_datetime(frame["日期"], errors="coerce").dt.normalize()
    frame["_day"] = day
    # 用 Period 而不是 day - timedelta：NaT 行不会触发 timedelta -> int64 的铸造警告
    frame["_week"] = day.dt.to_period("W-SUN").dt.start_time
    frame["_month"] = day.dt.to_period("M").dt.start_time
    return frame


def _aggregate(frame: pd.DataFrame, key: str, label: str) -> pd.DataFrame:
    grouped = (
        frame.dropna(subset=[key, DISTANCE, DURATION])
        .groupby(key)
        .agg(
            次数=(DISTANCE, "size"),
            里程=(DISTANCE, "sum"),
            时长=(DURATION, "sum"),
            平均配速=(PACE, "mean"),
        )
        .reset_index()
    )
    return grouped.rename(columns={key: label})


def volume_breakdown(df: pd.DataFrame):
    """返回 (周表, 月表)。列为：起始日 / 次数 / 里程 / 时长 / 平均配速。"""
    frame = add_time_keys(df)
    return _aggregate(frame, "_week", "周起始"), _aggregate(frame, "_month", "月份")


def volume_summary(df: pd.DataFrame) -> list:
    """跑量关键指标，直接返回 (标签, 显示值) 列表，页面照原样铺进表格。"""
    frame = add_time_keys(df).dropna(subset=["_day"])
    weekly, monthly = volume_breakdown(frame)
    if weekly.empty:
        return [("周跑量", "—")]

    km = weekly["里程"]
    last_four = weekly.tail(4)
    recent_month = frame["_month"].max()
    month_km = frame.loc[frame["_month"] == recent_month, DISTANCE].sum()
    span_days = int((frame["_day"].max() - frame["_day"].min()).days) + 1

    return [
        ("训练周数", "{0} 周".format(len(weekly))),
        ("周均跑量", "{0:.1f} km".format(km.mean())),
        ("最大单周", "{0:.1f} km".format(km.max())),
        ("最近 4 周", "{0:.1f} km".format(last_four["里程"].sum())),
        ("最近一个月", "{0:.1f} km（{1}）".format(month_km, recent_month.strftime("%Y-%m"))),
        ("训练跨度", "{0} 天 / {1} 月".format(span_days, len(monthly))),
    ]


# ---------------------------------------------------------------- 能力档案

def _confidence(count: int) -> float:
    """样本量置信度：30 次以上才算满信；记录越少，分数越向 50 收缩。"""
    return float(max(0.0, min(1.0, count / 30.0)))


def _shrink(score, confidence: float):
    """按置信度把极端分往中间拉：8 次记录拿到 100 分是误导，63 分才是它该有的样子。"""
    if score is None:
        return None
    return float(max(0.0, min(100.0, 50.0 + (score - 50.0) * confidence)))


def _clamp(value, low: float, high: float):
    """把数值线性映射到 0-100 分；None / 非有限值原样返回 None（页面显示「—」）。"""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    if high <= low:
        return 50.0
    return float(max(0.0, min(100.0, (number - low) / (high - low) * 100.0)))


def equivalent_5k_seconds(distance_km: float, duration_min: float):
    """Riegel 折算：T2 = T1 x (D2 / D1) ^ 1.06，把任意一次记录折算成等效 5 公里成绩。"""
    try:
        if not distance_km or float(distance_km) <= 0 or not duration_min:
            return None
        return float(duration_min) * 60.0 * (5.0 / float(distance_km)) ** 1.06
    except (TypeError, ValueError):
        return None


def estimate_vo2max(seconds_5k):
    """Daniels & Gilbert 跑步耗氧公式在等效 5 公里成绩上的近似值（不是实验室实测）。"""
    if not seconds_5k or seconds_5k <= 0:
        return None
    velocity = 5000.0 / (seconds_5k / 60.0)  # m/min
    vo2 = -4.60 + 0.182258 * velocity + 0.000104 * velocity ** 2
    return float(max(0.0, vo2))


def best_effort(df: pd.DataFrame):
    """最好的一次：只在单次 >= 3 km 的记录里挑等效 5 公里最快的，短距离外推误差太大。"""
    frame = df[df[DISTANCE] >= 3.0]
    if frame.empty:
        return None, None, None
    rows = []
    for _, row in frame.iterrows():
        seconds = equivalent_5k_seconds(row[DISTANCE], row[DURATION])
        if seconds:
            rows.append((seconds, row["日期"], row[DISTANCE]))
    if not rows:
        return None, None, None
    seconds, day, km = min(rows, key=lambda item: item[0])
    return seconds, day, km


def build_ability_profile(df: pd.DataFrame) -> list:
    """六个维度的规则评分。每项都给出分数、依据（真实数值）与口径（实测 / 代理 / 估算）。"""
    frame = add_time_keys(df).dropna(subset=["_day"])
    weekly, _ = volume_breakdown(df)
    profile = []

    # 1 心血管基础：每一跳能把多少米推出去（m/beat）。比「心率低于 150 的场次占比」公平：
    # 后者在慢跑为主的入门数据里恒为 100%，在冲刺计划里恒为 0%，量纲完全不同。
    if HEART_RATE_COLUMN in frame.columns and frame[HEART_RATE_COLUMN].notna().any():
        speed = frame[DISTANCE] * 1000.0 / frame[DURATION]
        beat = speed / frame[HEART_RATE_COLUMN]
        keep = beat[beat.notna() & np.isfinite(beat)]
        efficiency = float(keep.median()) if len(keep) else None
        if efficiency:
            share = float((frame[HEART_RATE_COLUMN] < 150).mean())
            profile.append({
                "name": "心血管基础",
                "score": _clamp(efficiency, 0.95, 1.25),
                "basis": "每跳推进 {0:.2f} 米（中位），心率 <150 的场次占 {1:.0%}".format(efficiency, share),
                "caliber": "实测",
            })
        else:
            profile.append({"name": "心血管基础", "score": None,
                            "basis": "心率与时长不足，无法判断", "caliber": "数据不足"})
    elif len(frame) >= 2 and frame[PACE].mean():
        cv = float(frame[PACE].std() / frame[PACE].mean())
        profile.append({
            "name": "心血管基础",
            # 配速变异系数 0.03（极稳）满分、0.12（忽快忽慢）零分；无心率只能看这个
            "score": _clamp(0.12 - cv, 0.0, 0.12),
            "basis": "无心率数据，用配速波动代理（变异系数 {0:.2f}）".format(cv),
            "caliber": "代理",
        })
    else:
        profile.append({"name": "心血管基础", "score": None,
                        "basis": "记录不足，无法判断", "caliber": "数据不足"})

    # 2 最大摄氧量：等效 5 公里 -> Daniels 公式
    seconds, day, km = best_effort(frame)
    vo2 = estimate_vo2max(seconds)
    if vo2:
        profile.append({
            "name": "最大摄氧量",
            # 跑步表估算的 VO2max 普遍低于实验室值（Riegel 外推 + 短距离样本），区间按跑步表实际分布取
            "score": _clamp(vo2, 20, 52),
            "basis": "最佳 {0:.1f} km 折算 5 公里 {1}，估算 {2:.0f} ml/kg/min".format(
                km, format_seconds(seconds), vo2),
            "caliber": "估算",
        })
    else:
        profile.append({"name": "最大摄氧量", "score": None,
                        "basis": "需要至少一次 3 km 以上的记录", "caliber": "数据不足"})

    # 3 耐力：单次最长 + 累计里程
    longest = float(frame[DISTANCE].max()) if len(frame) else None
    total = float(frame[DISTANCE].sum()) if len(frame) else None
    # 里程的增长是指数的（3 km -> 12 km 是四倍，20 km -> 60 km 不现实），线性打分会把所有人都挤在低分段
    long_score = _clamp(np.log2(max(1.0, longest)), 1.6, 4.2)
    total_score = _clamp(np.log10(max(1.0, total)), 1.3, 3.6)
    if long_score is not None and total_score is not None:
        profile.append({
            "name": "耐力",
            # 单次最长给 55%%：能连续跑多久是耐力本身；累计里程给 45%%：底子需要时间堆
            "score": long_score * 0.55 + total_score * 0.45,
            "basis": "单次最长 {0:.1f} km，累计 {1:.1f} km".format(longest, total),
            "caliber": "实测",
        })
    else:
        profile.append({"name": "耐力", "score": None, "basis": "无记录", "caliber": "数据不足"})

    # 4 速度：最快配速（越小越快）
    fastest = float(frame[PACE].min()) if len(frame) else None
    if fastest:
        profile.append({
            "name": "速度",
            "score": _clamp(8.0 - fastest, 0.0, 4.0),
            "basis": "最快配速 {0:.2f} min/km（{1}）".format(fastest, format_seconds(fastest * 60)),
            "caliber": "实测",
        })
    else:
        profile.append({"name": "速度", "score": None, "basis": "无记录", "caliber": "数据不足"})

    # 5 恢复质量：最长连跑天数。连着跑得越久，越说明没给身体留恢复窗口；
    # 只算间隔中位数会失效——样本里几乎所有数据的中位数都是 2 天，一律满分，没有区分度。
    days = frame["_day"].drop_duplicates().sort_values()
    if len(days) >= 2:
        ordinals = np.array([d.toordinal() for d in days])
        streak = current = 1
        for index in range(1, len(ordinals)):
            current = current + 1 if ordinals[index] - ordinals[index - 1] == 1 else 1
            streak = max(streak, current)
        span = int((days.iloc[-1] - days.iloc[0]).days) + 1
        rest = 1.0 - len(days) / float(max(1, span))
        profile.append({
            "name": "恢复质量",
            "score": _clamp(100.0 - 6.0 * max(0, streak - 1), 0.0, 100.0),
            "basis": "最长连跑 {0} 天，{1} 天里休息 {2} 天".format(streak, span, int(round(rest * span))),
            "caliber": "代理",
        })
    else:
        profile.append({"name": "恢复质量", "score": None,
                        "basis": "需要至少两天的记录", "caliber": "数据不足"})

    # 6 训练纪律：周跑量变异系数（节奏稳不稳）+ 有记录的周占比
    discipline = None
    basis = "无记录"
    if len(weekly) >= 2 and weekly["里程"].mean():
        cv = float(weekly["里程"].std() / weekly["里程"].mean())
        weeks = weekly["周起始"].sort_values()
        span = max(1, int(round((weeks.iloc[-1] - weeks.iloc[0]).days / 7)) + 1)
        continuity = min(1.0, len(weeks) / float(span))
        cv_score = _clamp(0.75 - cv, 0.0, 0.60)
        keep_score = _clamp(continuity, 0.40, 1.0)
        per_week = len(frame) / float(span)
        freq_score = _clamp(per_week, 1.5, 5.5)
        if cv_score is not None and keep_score is not None:
            discipline = cv_score * 0.4 + keep_score * 0.3 + freq_score * 0.3
            basis = "周跑量变异系数 {0:.2f}，{1} 周里有 {2} 周有记录，平均每周 {3:.1f} 次".format(
                cv, span, len(weeks), per_week)
    profile.append({"name": "训练纪律", "score": discipline,
                    "basis": basis, "caliber": "实测" if discipline is not None else "数据不足"})

    # 样本量收缩：记录越少，越不该给出满格或零格的极端结论
    confidence = _confidence(len(frame))
    for item in profile:
        raw = item["score"]
        item["raw"] = raw
        item["score"] = _shrink(raw, confidence)
        if raw is not None and confidence < 1.0:
            item["basis"] = "{0}；样本仅 {1} 次，评分已向中位收缩".format(item["basis"], len(frame))
    return profile


# ---------------------------------------------------------------- 负荷与表现

def build_load_series(df: pd.DataFrame, window_days: int = 30):
    """短期负荷 = 最近 7 天里程合计；长期负荷 = 最近 28 天合计折成周均。

    返回 DataFrame：date / short / long / ratio（ACWR）。记录不足时返回 None。
    """
    frame = add_time_keys(df).dropna(subset=["_day"])
    if frame.empty:
        return None
    daily = frame.groupby("_day")[DISTANCE].sum()
    index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(index, fill_value=0.0).astype(float)
    short = daily.rolling(7, min_periods=1).sum()
    long = daily.rolling(28, min_periods=7).sum() / 4.0
    table = pd.DataFrame({"date": index, "short": short.values, "long": long.values})
    table["ratio"] = table["short"] / table["long"].replace(0.0, np.nan)
    return table.tail(window_days).reset_index(drop=True)


def performance_trend(df: pd.DataFrame):
    """逐次估算 VO2max 与跑步表现指数（RPI，以第一次记录 = 100 为基准）。

    只取单次 >= 3 km 的记录：更短的距离外推到 5 公里误差太大，画出来只会误导。
    """
    frame = add_time_keys(df).dropna(subset=["_day"]).sort_values("_day")
    frame = frame[frame[DISTANCE] >= 3.0]
    rows = []
    for _, row in frame.iterrows():
        seconds = equivalent_5k_seconds(row[DISTANCE], row[DURATION])
        vo2 = estimate_vo2max(seconds)
        if vo2:
            rows.append({"date": row["_day"], "vo2": vo2, "seconds": seconds})
    if not rows:
        return pd.DataFrame(columns=["date", "vo2", "seconds", "rpi"])
    table = pd.DataFrame(rows)
    base = table["vo2"].iloc[0]
    table["rpi"] = table["vo2"] / base * 100.0 if base else np.nan
    return table


# ---------------------------------------------------------------- 单月复盘

def month_options(df: pd.DataFrame) -> list:
    """可选的自然月，按时间升序，格式 2026-10。"""
    frame = add_time_keys(df).dropna(subset=["_month"])
    if frame.empty:
        return []
    return [item.strftime("%Y-%m") for item in sorted(frame["_month"].unique())]


def slice_month(df: pd.DataFrame, label: str) -> pd.DataFrame:
    frame = add_time_keys(df).dropna(subset=["_month"])
    if frame.empty:
        return frame
    target = pd.to_datetime(label + "-01")
    return frame.loc[frame["_month"] == target].drop(columns=["_day", "_week", "_month"]).reset_index(drop=True)


def month_stats_text(df: pd.DataFrame, label: str) -> str:
    """给模型看的单月摘要：只描述这个月的真实数字，不做推断。"""
    frame = add_time_keys(df).dropna(subset=["_month"])
    target = pd.to_datetime(label + "-01")
    month = frame.loc[frame["_month"] == target]
    if month.empty:
        return "{0} 没有记录。".format(label)

    previous = frame.loc[frame["_month"] < target, "_month"].max()
    lines = [
        "统计月份：{0}".format(label),
        "训练次数：{0} 次，覆盖 {1} 天".format(len(month), month["_day"].nunique()),
        "月跑量：{0:.2f} km".format(month[DISTANCE].sum()),
        "月总时长：{0:.1f} 分钟".format(month[DURATION].sum()),
        "平均配速：{0:.2f} min/km（最快 {1:.2f}，最慢 {2:.2f}）".format(
            month[PACE].mean(), month[PACE].min(), month[PACE].max()),
        "单次距离：最长 {0:.2f} km，最短 {1:.2f} km".format(month[DISTANCE].max(), month[DISTANCE].min()),
    ]
    if HEART_RATE_COLUMN in month.columns and month[HEART_RATE_COLUMN].notna().any():
        lines.append("平均心率：{0:.0f} bpm（{1:.0f}–{2:.0f}）".format(
            month[HEART_RATE_COLUMN].mean(), month[HEART_RATE_COLUMN].min(), month[HEART_RATE_COLUMN].max()))
    else:
        lines.append("平均心率：该月无心率数据")

    inner, _ = volume_breakdown(month.drop(columns=["_day", "_week", "_month"]))
    if not inner.empty:
        lines.append("当月最大单周：{0:.1f} km（{1} 周起）".format(
            inner["里程"].max(), inner.loc[inner["里程"].idxmax(), "周起始"].strftime("%m-%d")))

    if pd.notna(previous):
        before = frame.loc[frame["_month"] == previous]
        if len(before):
            delta = month[DISTANCE].sum() - before[DISTANCE].sum()
            lines.append("与上个有记录的月（{0}）相比：跑量 {1:+.1f} km，次数 {2:+d}".format(
                previous.strftime("%Y-%m"), delta, len(month) - len(before)))
    return "\n".join(lines)


def month_records_csv(df: pd.DataFrame, label: str) -> str:
    """该月原始记录（只保留四列 + 心率），直接喂给模型做逐条引用。"""
    month = slice_month(df, label)
    if month.empty:
        return ""
    keep = [c for c in ("日期", DISTANCE, DURATION, PACE, HEART_RATE_COLUMN) if c in month.columns]
    return month[keep].to_csv(index=False)