from __future__ import annotations

import pandas as pd


def is_risk_name(name: str) -> bool:
    upper = str(name).upper()
    return "ST" in upper or "*ST" in upper or "退" in str(name)


def is_limit_up(row: pd.Series, limit_up_pct: float = 0.095) -> bool:
    prev_close = row.get("prev_close")
    if pd.isna(prev_close) or prev_close <= 0:
        return False
    return (row["close"] / prev_close - 1) >= limit_up_pct


def is_long_upper_shadow(row: pd.Series, rules: dict) -> bool:
    body = abs(float(row["close"]) - float(row["open"]))
    body = max(body, 0.01)
    upper_shadow = float(row["high"]) - max(float(row["open"]), float(row["close"]))
    avg_amount = float(row.get("avg_amount_20d", 0) or 0)
    if avg_amount <= 0:
        return False
    return (
        upper_shadow / body > rules.get("long_upper_shadow_ratio", 1.5)
        and float(row["amount"]) > avg_amount * rules.get("long_upper_shadow_amount_multiple", 1.5)
        and (float(row["high"]) - float(row["close"])) / float(row["high"])
        > rules.get("long_upper_shadow_close_below_high", 0.03)
    )


def has_three_day_huge_amount(df: pd.DataFrame, multiple: float = 2.0, days: int = 3) -> bool:
    if len(df) < days:
        return False
    tail = df.tail(days)
    return bool((tail["amount"] > tail["avg_amount_20d"] * multiple).all())


def evaluate_hard_filters(
    symbol: str,
    name: str,
    df: pd.DataFrame,
    settings: dict,
    rules: dict,
) -> list[str]:
    reasons: list[str] = []
    if is_risk_name(name):
        reasons.append("名称包含 ST、*ST 或退市风险字样")
    if df.empty or len(df) < rules.get("min_history_days", 120):
        reasons.append("最近120个交易日数据不足")
        return reasons

    work = df.copy()
    work["prev_close"] = work["close"].shift(1)
    latest = work.iloc[-1]
    if pd.isna(latest.get("amount")) or latest.get("amount", 0) <= 0:
        reasons.append("停牌或最新成交额为空")
    if latest.get("avg_amount_20d", 0) < settings["min_avg_amount_20d"]:
        reasons.append("20日平均成交额低于3亿元")
    if latest["close"] < latest.get("ma60", float("inf")):
        reasons.append("收盘价低于MA60")
    if is_limit_up(latest, rules.get("limit_up_pct", 0.095)):
        reasons.append("当日涨停或接近涨停")
    if latest.get("return_5d", 0) > rules.get("max_5d_return", 0.15):
        reasons.append("近5日涨幅大于15%")
    if latest.get("return_10d", 0) > rules.get("max_10d_return", 0.25):
        reasons.append("近10日涨幅大于25%")
    if latest.get("ma20_deviation", 0) > rules.get("max_ma20_deviation", 0.12):
        reasons.append("收盘价距离MA20超过12%")
    if is_long_upper_shadow(latest, rules):
        reasons.append("当日放量长上影")
    if has_three_day_huge_amount(
        work,
        rules.get("huge_amount_multiple", 2.0),
        rules.get("huge_amount_days", 3),
    ):
        reasons.append("连续3日成交额高于20日均额2倍")
    if latest["close"] < latest.get("ma20", float("inf")):
        reasons.append("收盘价跌破MA20")
    return reasons
