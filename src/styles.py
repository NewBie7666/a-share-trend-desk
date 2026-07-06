from __future__ import annotations

from collections import defaultdict

import pandas as pd


STYLE_DISPLAY_NAMES = {
    "finance_brokerage": "券商",
    "finance_bank": "银行",
    "finance_insurance": "保险",
    "semiconductor": "半导体",
    "technology_hardware": "科技硬件",
    "consumer": "消费",
    "medicine": "医药",
    "medicine_innovation_drug": "创新药",
    "medicine_cxo": "医药CXO",
    "medicine_tcm": "中药",
    "medicine_device": "医疗器械",
    "medicine_pharmacy": "医药商业",
    "resource": "资源",
    "high_dividend": "高股息",
    "manufacturing": "制造",
    "other": "其他",
}

STYLE_RISK_NOTES = {
    "finance_brokerage": "券商股高度依赖市场成交额、指数风险偏好和政策预期，若指数转弱需快速减仓",
    "finance_bank": "银行股受利率、息差、资产质量和宏观信用周期影响，弹性通常弱于券商",
    "finance_insurance": "保险股受权益市场、长端利率、负债端销售和资产质量影响",
    "semiconductor": "半导体波动较高，受周期、库存、国产替代预期和估值切换影响",
    "technology_hardware": "科技硬件受订单、景气度和估值波动影响，需警惕高位放量回落",
    "consumer": "消费股受需求恢复、品牌竞争和估值中枢影响",
    "medicine": "医药股受政策、集采、研发进展和估值修复预期影响",
    "medicine_innovation_drug": "创新药受研发进展、出海授权、医保谈判、临床数据和估值修复影响，波动可能明显放大",
    "medicine_cxo": "CXO受海外订单、投融资周期、地缘政策和行业景气度影响，趋势反转需观察订单验证",
    "medicine_tcm": "中药股受政策扶持、品牌提价、业绩兑现和估值切换影响，弹性通常弱于创新药",
    "medicine_device": "医疗器械受集采、医院采购、国产替代和新品放量影响，需警惕政策压价",
    "medicine_pharmacy": "医药商业和药房受处方外流、门店效率、医保控费和竞争格局影响",
    "resource": "资源股受商品价格、供需周期和汇率影响，波动可能较大",
    "high_dividend": "高股息资产受利率、分红预期和防御偏好影响，趋势弹性通常有限",
    "manufacturing": "制造股受订单周期、成本、出口和行业景气影响",
    "other": "风格未知，需人工确认后再考虑交易",
}

STYLE_KEYWORDS = {
    "finance_brokerage": ["证券", "国泰海通", "中信建投", "东方财富"],
    "finance_bank": ["银行"],
    "finance_insurance": ["保险", "中国平安"],
    "semiconductor": ["半导体", "芯片", "微电", "长电", "士兰", "兆易", "紫光国微", "韦尔", "华天", "通富"],
    "technology_hardware": ["科技", "电子", "光电", "通信", "电路", "浪潮", "曙光", "工业富联", "沪电", "深南", "京东方", "TCL"],
    "consumer": ["茅台", "五粮液", "泸州", "伊利", "美的", "格力"],
    "medicine_innovation_drug": ["创新药", "恒瑞", "百济", "信达", "君实", "康方", "泽璟", "贝达", "荣昌", "迪哲"],
    "medicine_cxo": ["药明康德", "康德", "凯莱英", "泰格", "昭衍", "美迪西", "博腾"],
    "medicine_tcm": ["白药", "片仔癀", "同仁堂", "以岭", "中药", "云南白药"],
    "medicine_device": ["医疗器械", "迈瑞", "联影", "鱼跃", "安图", "万孚", "开立"],
    "medicine_pharmacy": ["药房", "大参林", "老百姓", "益丰", "一心堂", "医药商业"],
    "medicine": ["医药", "药"],
    "resource": ["矿", "黄金", "稀土", "钨", "锂", "铜", "铝", "钼", "锗", "锌"],
    "high_dividend": ["神华", "长江电力", "大唐发电"],
    "manufacturing": ["重工", "玻璃", "化学", "建筑", "船舶", "三花", "比亚迪"],
}

SYMBOL_STYLE_OVERRIDES = {
    "600030": "finance_brokerage",
    "601211": "finance_brokerage",
    "601688": "finance_brokerage",
    "600999": "finance_brokerage",
    "601066": "finance_brokerage",
    "000776": "finance_brokerage",
    "000783": "finance_brokerage",
    "600036": "finance_bank",
    "601166": "finance_bank",
    "601398": "finance_bank",
    "601318": "finance_insurance",
    "601336": "finance_insurance",
    "600276": "medicine_innovation_drug",
    "600703": "semiconductor",
}


def identify_best_style(symbol: str, name: str) -> str:
    symbol = str(symbol).zfill(6)
    if symbol in SYMBOL_STYLE_OVERRIDES:
        return SYMBOL_STYLE_OVERRIDES[symbol]
    text = str(name)
    for style, keywords in STYLE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return style
    return "other"


def style_state_from_score(score: float) -> str:
    if score >= 75:
        return "strong"
    if score >= 55:
        return "neutral"
    return "weak"


def style_reason(style: str) -> str:
    display = STYLE_DISPLAY_NAMES.get(style, style)
    if style == "other":
        return "风格未知，需人工确认"
    return f"{display}风格强势，个股成交活跃，相对沪深300走强"


def style_risk_note(style: str) -> str:
    return STYLE_RISK_NOTES.get(style, STYLE_RISK_NOTES["other"])


def _style_rotation_metrics(latest_rows: list[pd.Series]) -> dict:
    total = len(latest_rows)
    if not total:
        return {
            "breadth_score": 0.0,
            "style_momentum_score": 0.0,
            "style_retreat_penalty": 0.0,
            "style_rotation_bonus": 0.0,
            "retreat_risk": False,
            "retreat_ratio": 0.0,
        }
    above_ma20 = sum(row["close"] > row.get("ma20", float("inf")) for row in latest_rows) / total
    above_ma60 = sum(row["close"] > row.get("ma60", float("inf")) for row in latest_rows) / total
    rs_positive = sum(row.get("relative_strength_20d", 0) > 0 for row in latest_rows) / total
    macd_bull = sum(row.get("macd_dif", 0) > row.get("macd_dea", 0) for row in latest_rows) / total
    avg_5d = sum(float(row.get("return_5d", 0) or 0) for row in latest_rows) / total
    avg_10d = sum(float(row.get("return_10d", 0) or 0) for row in latest_rows) / total
    retreat_count = sum(
        bool(
            row.get("close", 0) < row.get("ma5", row.get("close", 0))
            or row.get("return_5d", 0) < -0.03
            or row.get("macd_dif", 0) < row.get("macd_dea", 0)
        )
        for row in latest_rows
    )
    retreat_ratio = retreat_count / total
    breadth_score = above_ma20 * 60 + rs_positive * 40
    momentum_score = (
        above_ma20 * 25
        + above_ma60 * 20
        + rs_positive * 25
        + macd_bull * 20
        + max(min(avg_5d, 0.12), -0.12) / 0.12 * 5
        + max(min(avg_10d, 0.18), -0.18) / 0.18 * 5
    )
    retreat_penalty = 15.0 if retreat_ratio >= 0.50 and above_ma20 < 0.65 else 8.0 if retreat_ratio >= 0.35 else 0.0
    rotation_bonus = 8.0 if above_ma20 >= 0.60 and rs_positive >= 0.60 and macd_bull >= 0.50 and avg_5d > 0 else 0.0
    return {
        "breadth_score": round(max(0.0, min(100.0, breadth_score)), 2),
        "style_momentum_score": round(max(0.0, min(100.0, momentum_score)), 2),
        "style_retreat_penalty": round(retreat_penalty, 2),
        "style_rotation_bonus": round(rotation_bonus, 2),
        "retreat_risk": retreat_penalty > 0,
        "retreat_ratio": round(retreat_ratio, 4),
    }


def build_style_state_table(
    stock_frames: dict[str, pd.DataFrame],
    names: dict[str, str],
    confirmed_symbols: set[str] | None = None,
    fetch_results_by_symbol: dict | None = None,
) -> list[dict]:
    grouped: dict[str, list[pd.Series]] = defaultdict(list)
    for symbol, df in stock_frames.items():
        if confirmed_symbols is not None and symbol not in confirmed_symbols:
            continue
        if df.empty:
            continue
        style = identify_best_style(symbol, names.get(symbol, ""))
        if style == "other":
            continue
        grouped[style].append(df.iloc[-1])

    rows = []
    for style, latest_rows in sorted(grouped.items()):
        total = len(latest_rows)
        above_ma20 = sum(row["close"] > row.get("ma20", float("inf")) for row in latest_rows) / total
        above_ma60 = sum(row["close"] > row.get("ma60", float("inf")) for row in latest_rows) / total
        rs_positive = sum(row.get("relative_strength_20d", 0) > 0 for row in latest_rows) / total
        macd_bull = sum(row.get("macd_dif", 0) > row.get("macd_dea", 0) for row in latest_rows) / total
        composite = (above_ma20 + above_ma60 + rs_positive + macd_bull) / 4
        rotation = _style_rotation_metrics(latest_rows)
        style_symbols = [
            symbol
            for symbol, df in stock_frames.items()
            if not df.empty
            and (confirmed_symbols is None or symbol in confirmed_symbols)
            and identify_best_style(symbol, names.get(symbol, "")) == style
        ]
        cache_count = 0
        failed_count = 0
        if fetch_results_by_symbol:
            for symbol in style_symbols:
                result = fetch_results_by_symbol.get(symbol)
                if result is None:
                    continue
                if getattr(result, "final_source", "fresh") == "cache":
                    cache_count += 1
                elif getattr(result, "final_source", "fresh") == "failed" or not getattr(result, "is_expected_trade_date", True):
                    failed_count += 1
        cache_ratio = cache_count / total if total else 0.0
        failed_ratio = failed_count / total if total else 0.0
        state = style_state_from_score(composite * 100)
        confidence = state_confidence(total)
        if cache_ratio > 0.50:
            if state == "strong":
                state = "neutral"
            confidence = "low"
        elif cache_ratio > 0.20 or failed_ratio > 0.20:
            confidence = lower_confidence(confidence)
        sample_quality_eligible = (
            total >= 5
            and confidence in {"medium", "high"}
            and cache_ratio <= 0.20
            and failed_ratio <= 0.05
        )
        strongest_eligible = state == "strong" and sample_quality_eligible
        rows.append(
            {
                "style": style,
                "display_name": STYLE_DISPLAY_NAMES.get(style, style),
                "state": state,
                "sample_size": total,
                "state_confidence": confidence,
                "cache_count": cache_count,
                "cache_ratio": round(cache_ratio, 4),
                "failed_count": failed_count,
                "failed_ratio": round(failed_ratio, 4),
                "sample_quality_eligible": sample_quality_eligible,
                "strongest_eligible": strongest_eligible,
                "above_ma20_ratio": round(above_ma20, 4),
                "above_ma60_ratio": round(above_ma60, 4),
                "rs_positive_ratio": round(rs_positive, 4),
                "macd_bull_ratio": round(macd_bull, 4),
                "breadth_score": rotation["breadth_score"],
                "style_momentum_score": rotation["style_momentum_score"],
                "style_rotation_bonus": rotation["style_rotation_bonus"],
                "style_retreat_penalty": rotation["style_retreat_penalty"],
                "retreat_risk": rotation["retreat_risk"],
                "retreat_ratio": rotation["retreat_ratio"],
            }
        )
    return rows


def build_style_state_table_from_snapshots(
    snapshots_by_symbol: dict,
    confirmed_symbols: set[str] | None = None,
) -> list[dict]:
    stock_frames = {
        symbol: snapshot.indicators
        for symbol, snapshot in snapshots_by_symbol.items()
        if getattr(snapshot, "indicators", pd.DataFrame()) is not None
        and not getattr(snapshot, "indicators", pd.DataFrame()).empty
    }
    names = {
        symbol: getattr(snapshot, "name", "")
        for symbol, snapshot in snapshots_by_symbol.items()
    }
    fetch_results_by_symbol = {
        symbol: getattr(snapshot, "fetch_result", None)
        for symbol, snapshot in snapshots_by_symbol.items()
        if getattr(snapshot, "fetch_result", None) is not None
    }
    return build_style_state_table(stock_frames, names, confirmed_symbols, fetch_results_by_symbol)


def state_confidence(sample_size: int) -> str:
    if sample_size >= 8:
        return "high"
    if sample_size >= 5:
        return "medium"
    return "low"


def lower_confidence(confidence: str) -> str:
    return {"high": "medium", "medium": "low", "low": "low"}.get(confidence, "low")
