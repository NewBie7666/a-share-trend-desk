from __future__ import annotations

import pandas as pd

from .position_rules import candidate_min_score, get_position_rule
from .scoring import score_stock
from .snapshots import snapshot_fetch_results, snapshot_filter_reasons, snapshot_names, snapshots_to_frames
from .styles import STYLE_DISPLAY_NAMES
from .trading_decision import make_final_trade_decision
from .utils import round_price


TREND_STYLES = {"technology_hardware", "semiconductor", "resource", "manufacturing"}


def one_lot_amount(price: float, settings: dict) -> float:
    return float(price) * int(settings.get("lot_size", 100))


def current_holding_market_value(holdings: pd.DataFrame | None, stock_frames: dict[str, pd.DataFrame]) -> float:
    if holdings is None or holdings.empty:
        return 0.0
    total = 0.0
    for _, holding in holdings.iterrows():
        symbol = str(holding.get("symbol", "")).zfill(6)
        df = stock_frames.get(symbol)
        if df is None or df.empty:
            continue
        total += float(holding.get("quantity", 0) or 0) * float(df.iloc[-1]["close"])
    return total


def build_condition_order(candidate: dict, settings: dict, rules: dict, rank: int, raw_amount: float) -> dict:
    order_rules = rules["condition_order"]
    close = float(candidate["close_price"])
    ma20 = float(candidate["ma20"])
    ma60 = float(candidate["ma60"])
    trigger = max(close * order_rules["trigger_close_multiple"], ma20 * order_rules["trigger_ma20_multiple"])
    half_stop = trigger * 0.92
    full_stop = trigger * 0.88
    trend_stop = ma60 * 0.99
    final_exit = max(full_stop, trend_stop)
    lot_size = int(settings.get("lot_size", 100))
    lot_amount = trigger * lot_size
    suggested_lots = int(raw_amount // lot_amount) if lot_amount > 0 else 0
    suggested_shares = suggested_lots * lot_size
    estimated_amount = suggested_shares * trigger
    account_risk_pct = 0.0
    if settings["initial_cash"] > 0 and trigger > 0:
        account_risk_pct = estimated_amount * (trigger - final_exit) / trigger / settings["initial_cash"]
    return {
        "candidate_rank": rank,
        "symbol": candidate["symbol"],
        "name": candidate["name"],
        "rank": rank,
        "score": candidate["score"],
        "base_signal_score": candidate.get("base_signal_score", candidate.get("score")),
        "signal_score": candidate.get("signal_score", candidate["score"]),
        "style_momentum_score": candidate.get("style_momentum_score", ""),
        "breadth_score": candidate.get("breadth_score", ""),
        "style_rotation_bonus": candidate.get("style_rotation_bonus", ""),
        "style_retreat_penalty": candidate.get("style_retreat_penalty", ""),
        "retreat_risk": candidate.get("retreat_risk", False),
        "structure_state": candidate.get("structure_state", ""),
        "entry_quality_score": candidate.get("entry_quality_score", ""),
        "decision_confidence": candidate.get("decision_confidence", ""),
        "timing_decision": candidate.get("timing_decision", ""),
        "timing_reason": candidate.get("timing_reason", ""),
        "trend_strong": candidate.get("trend_strong", ""),
        "timing_risk_tag": candidate.get("timing_risk_tag", ""),
        "final_action": candidate.get("final_action", ""),
        "decision_source": candidate.get("decision_source", ""),
        "position_multiplier": candidate.get("position_multiplier", ""),
        "final_reason": candidate.get("final_reason", ""),
        "market_strength": candidate.get("market_strength", 0),
        "market_regime": candidate.get("market_regime", ""),
        "market_score": candidate.get("market_score", ""),
        "data_integrity_score": candidate.get("data_integrity_score", 1.0),
        "confidence_position_multiplier": candidate.get("confidence_position_multiplier", 1.0),
        "final_score": candidate["final_score"],
        "best_style": candidate["best_style"],
        "style_state": candidate["style_state"],
        "style_scores": candidate["style_scores"],
        "close_price": round_price(close),
        "trigger_price": round_price(trigger),
        "suggested_buy_amount": round(estimated_amount, 2),
        "suggested_lots": suggested_lots,
        "suggested_shares": suggested_shares,
        "estimated_amount": round(estimated_amount, 2),
        "half_stop_price": round_price(half_stop),
        "full_stop_price": round_price(full_stop),
        "trend_stop_price": round_price(trend_stop),
        "final_exit_price": round_price(final_exit),
        "half_stop_action": "\u8dcc\u7834 half_stop_price\uff0c\u51cf\u534a",
        "final_exit_action": "\u8dcc\u7834 final_exit_price\uff0c\u6e05\u4ed3",
        "trend_exit_action": "\u8fde\u7eed2\u65e5\u6536\u76d8\u8dcc\u7834MA60\uff0c\u6e05\u4ed3",
        "account_risk_pct": round(account_risk_pct, 4),
        "account_risk_note": "\u4ee5 final_exit_price \u4f5c\u4e3a\u6e05\u4ed3\u4ef7\u4f30\u7b97\u8d26\u6237\u6700\u5927\u5355\u7b14\u4e8f\u635f\u6bd4\u4f8b",
        "take_profit_reduce_price": round_price(trigger * order_rules["take_profit_reduce_multiple"]),
        "take_profit_strong_price": round_price(trigger * order_rules["take_profit_strong_multiple"]),
        "max_position_amount": round(raw_amount, 2),
        "reason": candidate["reason"],
        "risk_note": f"{candidate['risk_note']}\uff1bA\u80a1\u4e00\u624b\u4e3a100\u80a1\uff0c\u6309\u89e6\u53d1\u4ef7\u4f30\u7b97\u4e00\u624b\u7ea6 {lot_amount:.2f} \u5143",
    }


def _candidate_source_fields(symbol: str, fetch_results_by_symbol: dict | None) -> dict:
    result = (fetch_results_by_symbol or {}).get(symbol)
    if result is None:
        return {
            "candidate_data_source": "fresh",
            "candidate_latest_date": "",
            "candidate_raw_latest_date": "",
            "is_expected_trade_date": True,
            "data_source_name": "",
            "fallback_source_used": "",
            "source_attempts_summary": "",
            "source_error": "",
            "source_error_category": "",
        }
    return {
        "candidate_data_source": getattr(result, "final_source", "fresh"),
        "candidate_latest_date": getattr(result, "effective_latest_date", "") or getattr(result, "latest_date", "") or "",
        "candidate_raw_latest_date": getattr(result, "raw_latest_date", "") or "",
        "is_expected_trade_date": bool(getattr(result, "is_expected_trade_date", True)),
        "data_source_name": getattr(result, "data_source_name", "") or "",
        "fallback_source_used": getattr(result, "fallback_source_used", "") or "",
        "source_attempts_summary": _source_attempts_summary(getattr(result, "source_attempts", None)),
        "source_error": getattr(result, "error", "") or getattr(result, "data_staleness_reason", "") or "",
        "source_error_category": getattr(result, "error_category", "") or "",
    }


def _source_attempts_summary(attempts: list[dict] | None) -> str:
    if not attempts:
        return "无来源尝试记录"
    parts = []
    for item in attempts:
        status = "成功" if item.get("status") == "success" else "失败"
        parts.append(f"{item.get('source_name', '')}:{status},有效日={item.get('effective_latest_date') or '-'},类别={item.get('error_category') or '无错误'}")
    return "；".join(parts)


def _latest_trading_volume_check(symbol: str, stock_frames: dict[str, pd.DataFrame]) -> dict:
    df = stock_frames.get(symbol)
    if df is None or df.empty:
        return {"has_trading_volume": False, "volume": 0.0, "amount": 0.0, "reason": "\u65e5\u7ebf\u884c\u60c5\u6570\u636e\u4e3a\u7a7a"}
    latest = df.iloc[-1]
    if "volume" not in latest.index or "amount" not in latest.index:
        return {"has_trading_volume": True, "volume": 1.0, "amount": 1.0, "reason": ""}
    volume = float(latest.get("volume", 0) or 0)
    amount = float(latest.get("amount", 0) or 0)
    if volume <= 0 or amount <= 0:
        return {"has_trading_volume": False, "volume": volume, "amount": amount, "reason": "\u5019\u9009\u80a1\u6700\u65b0\u6210\u4ea4\u91cf\u6216\u6210\u4ea4\u989d\u4e3a0\uff0c\u505c\u724c\u98ce\u9669\u515c\u5e95\u672a\u901a\u8fc7"}
    return {"has_trading_volume": True, "volume": volume, "amount": amount, "reason": ""}


def _latest_has_trading_volume(symbol: str, stock_frames: dict[str, pd.DataFrame]) -> tuple[bool, str]:
    check = _latest_trading_volume_check(symbol, stock_frames)
    return bool(check["has_trading_volume"]), str(check.get("reason", ""))


def _data_issue_list(fetch_results_by_symbol: dict | None, names: dict[str, str]) -> list[dict]:
    rows = []
    for symbol, result in (fetch_results_by_symbol or {}).items():
        data = getattr(result, "data", pd.DataFrame())
        if getattr(result, "final_source", "fresh") == "failed" or data.empty:
            rows.append(
                {
                    "symbol": symbol,
                    "name": getattr(result, "name", names.get(symbol, "")),
                    "failed_reason": getattr(result, "error", "") or getattr(result, "data_staleness_reason", "") or "\u65e5\u7ebf\u884c\u60c5\u6570\u636e\u4e0d\u53ef\u7528",
                    "error_category": getattr(result, "error_category", "") or "unknown",
                }
            )
    return rows


def _normalize_observation(item: dict) -> dict:
    out = item.copy()
    out.setdefault("candidate_data_source", "not_reviewed")
    out.setdefault("candidate_latest_date", "-")
    out.setdefault("candidate_raw_latest_date", "-")
    out.setdefault("is_expected_trade_date", "-")
    out.setdefault("data_source_name", "")
    out.setdefault("fallback_source_used", "")
    out.setdefault("source_attempts_summary", "")
    out.setdefault("review_scope", "observation_only")
    out.setdefault("mode_account_risk_limit", "")
    out.setdefault("account_risk_pass", "")
    out.setdefault("structure_state", "")
    out.setdefault("entry_quality_score", "")
    out.setdefault("decision_confidence", "")
    out.setdefault("timing_decision", "")
    out.setdefault("timing_reason", "")
    out.setdefault("trend_strong", "")
    out.setdefault("timing_risk_tag", "")
    out.setdefault("final_action", "")
    out.setdefault("position_multiplier", "")
    out.setdefault("final_reason", "")
    out.setdefault("decision_source", "")
    out.setdefault("market_regime", "")
    out.setdefault("market_score", "")
    out.setdefault("portfolio_mode", "")
    out.setdefault("final_portfolio_mode", "")
    out.setdefault("raw_portfolio_mode", "")
    out.setdefault("volume", "")
    out.setdefault("amount", "")
    out.setdefault("minimum_score", "")
    out.setdefault("gate_pass", False)
    out.setdefault("gate_failed_reasons", [])
    out.setdefault("gate_checked_fields", {})
    out.setdefault("gate_checked_fields_json", "")
    out.setdefault("blocker_codes", [])
    out.setdefault("blocker_source", "")
    out.setdefault("blocker_severity", "")
    out.setdefault("final_trade_permission", "")
    out.setdefault("final_trade_permission_reason", "")
    return out


def _apply_timing_fields(item: dict, timing: dict | None) -> dict:
    if not timing:
        return item
    item.update(
        {
            "structure_state": timing.get("structure_state", ""),
            "entry_quality_score": timing.get("entry_quality_score", ""),
            "decision_confidence": timing.get("decision_confidence", ""),
            "timing_decision": timing.get("timing_decision", ""),
            "timing_reason": timing.get("timing_reason", ""),
            "trend_strong": timing.get("trend_strong", ""),
            "timing_risk_tag": timing.get("timing_risk_tag", ""),
        }
    )
    return item


def _apply_final_decision_fields(item: dict, decision: dict, market: dict) -> dict:
    portfolio = market.get("portfolio_mode", {}) or {}
    item["final_action"] = decision.get("final_action", "")
    item["position_multiplier"] = decision.get("position_multiplier", "")
    item["final_reason"] = "；".join(str(reason) for reason in decision.get("final_reason", []) if reason)
    item["decision_source"] = decision.get("decision_source", "")
    item["market_regime"] = market.get("market_regime", "")
    item["market_score"] = market.get("market_score", "")
    item["portfolio_mode"] = portfolio.get("portfolio_mode", "")
    item["final_portfolio_mode"] = portfolio.get("final_portfolio_mode", portfolio.get("portfolio_mode", ""))
    item["raw_portfolio_mode"] = portfolio.get("raw_portfolio_mode", portfolio.get("portfolio_mode", ""))
    return item


def classify_trading_critical_blockers(settings: dict) -> dict:
    reasons = [str(item) for item in settings.get("trading_critical_reasons", []) or []]
    structured_codes = [str(code) for code in (settings.get("trading_critical_blocker_codes", []) or []) if code]
    fallback_codes: list[str] = []
    if settings.get("trade_calendar_status") == "failed":
        structured_codes.append("trade_calendar_failed")
    if settings.get("stock_basic_is_trading_critical"):
        structured_codes.append("stock_basic_trading_critical")
    if settings.get("data_quality_level") == "stale":
        structured_codes.append("stale_data")

    if not structured_codes:
        text = "?".join(reasons)
        keyword_map = {
            "core_index_unavailable": ["核心指数", "core index", "000300", "000001", "399001"],
            "trade_calendar_failed": ["交易日历失败", "trade calendar failed"],
            "stock_basic_trading_critical": ["ST/退市", "主板范围无法判断", "股票基础信息为交易关键异常", "stock basic"],
            "market_state_unknown": ["市场状态不可判定", "market_state unknown", "market state unknown"],
            "broad_daily_data_failure": ["大面积个股日线异常", "大量缓存回退", "broad daily", "too many"],
            "stale_data": ["数据陈旧", "stale"],
        }
        for code, keywords in keyword_map.items():
            if any(keyword in text for keyword in keywords):
                fallback_codes.append(code)

    if structured_codes:
        blocker_source = "structured_code"
        codes = structured_codes
    elif fallback_codes:
        blocker_source = "fallback_keyword"
        codes = fallback_codes
    else:
        blocker_source = "none"
        codes = []

    unique_codes = sorted(set(code for code in codes if code))
    blocked = bool(unique_codes) and settings.get("data_quality_level") in {"critical_cache", "stale"}
    return {
        "blocked": blocked,
        "blocker_codes": unique_codes,
        "blocker_reasons": reasons,
        "blocker_source": blocker_source,
        "blocker_severity": "block_new_position" if blocked else "warning" if unique_codes else "none",
    }


def validate_pre_order_candidate(candidate: dict, market: dict, source_fields: dict, volume_check: dict, settings: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    portfolio = market.get("portfolio_mode", {}) or {}
    portfolio_mode = portfolio.get("portfolio_mode", "cash")
    market_regime = market.get("market_regime")
    expected_date = settings.get("expected_trade_date", "")
    timing_decision = candidate.get("timing_decision")
    final_action = candidate.get("final_action")
    data_source = source_fields.get("candidate_data_source")
    latest_date = source_fields.get("candidate_latest_date")

    if portfolio_mode == "cash":
        reasons.append("组合模式为 cash，禁止正式新开仓")
    if not market_regime:
        reasons.append("缺少 market_regime，不能进入正式候选")
    elif market_regime == "cash":
        reasons.append("市场环境为 cash，禁止正式新开仓")
    if not timing_decision:
        reasons.append("缺少 timing_decision，不能进入正式候选")
    elif timing_decision != "BUY":
        reasons.append("timing_decision 不是 BUY")
    if not final_action:
        reasons.append("缺少 final_action，不能进入正式候选")
    elif final_action != "BUY":
        reasons.append("final_action 不是 BUY")
    if not data_source:
        reasons.append("缺少 candidate_data_source，不能进入正式候选")
    elif data_source != "fresh":
        reasons.append("candidate_data_source 不是 fresh")
    if expected_date:
        if not latest_date:
            reasons.append("缺少 candidate_latest_date，不能进入正式候选")
        elif latest_date != expected_date:
            reasons.append("candidate_latest_date 不是预期交易日")
    if not source_fields.get("is_expected_trade_date"):
        reasons.append("候选行情未通过预期交易日校验")
    if not volume_check.get("has_trading_volume"):
        reasons.append(volume_check.get("reason") or "候选成交量/成交额复核未通过")
    if float(volume_check.get("volume", 0) or 0) <= 0:
        reasons.append("候选最新成交量为0")
    if float(volume_check.get("amount", 0) or 0) <= 0:
        reasons.append("候选最新成交额为0")
    return not reasons, reasons


def validate_final_candidate(candidate: dict, order: dict, market: dict, source_fields: dict, volume_check: dict, settings: dict) -> tuple[bool, list[str]]:
    pre_ok, reasons = validate_pre_order_candidate(candidate, market, source_fields, volume_check, settings)
    if not pre_ok:
        return False, reasons
    if order.get("account_risk_pass") is not True:
        reasons.append("账户风险未通过当前模式上限")
    if int(order.get("suggested_lots", 0) or 0) < 1:
        reasons.append("不足一手，不能生成正式买入候选")
    return not reasons, reasons


def build_candidate_gate_audit(
    candidate: dict,
    order: dict | None,
    market: dict,
    source_fields: dict,
    volume_check: dict,
    settings: dict,
    *,
    pre_validation: tuple[bool, list[str]] | None = None,
    final_validation: tuple[bool, list[str]] | None = None,
) -> dict:
    """Record the actual candidate gate result without introducing a second gate."""
    source_fields = source_fields or {}
    volume_check = volume_check or {}
    portfolio = (market.get("portfolio_mode", {}) or {}).get("portfolio_mode", "")
    checked_fields = {
        "portfolio_mode": portfolio,
        "market_regime": market.get("market_regime"),
        "timing_decision": candidate.get("timing_decision"),
        "final_action": candidate.get("final_action"),
        "candidate_data_source": source_fields.get("candidate_data_source"),
        "candidate_latest_date": source_fields.get("candidate_latest_date"),
        "expected_trade_date": settings.get("expected_trade_date"),
        "is_expected_trade_date": source_fields.get("is_expected_trade_date"),
        "volume": volume_check.get("volume"),
        "amount": volume_check.get("amount"),
        "has_trading_volume": volume_check.get("has_trading_volume"),
        "account_risk_pass": order.get("account_risk_pass") if order is not None else None,
        "suggested_lots": order.get("suggested_lots") if order is not None else None,
    }
    if final_validation is not None:
        gate_pass, reasons = final_validation
    elif pre_validation is not None:
        gate_pass, reasons = pre_validation
    elif order is None:
        gate_pass, reasons = validate_pre_order_candidate(candidate, market, source_fields, volume_check, settings)
    else:
        gate_pass, reasons = validate_final_candidate(candidate, order, market, source_fields, volume_check, settings)
    return {
        "gate_pass": bool(gate_pass),
        "gate_failed_reasons": list(reasons),
        "gate_checked_fields": checked_fields,
    }


def _candidate_review_row(candidate: dict, *, review_scope: str, downgrade_reasons: list[str], source_fields: dict | None = None, volume_check: dict | None = None, order: dict | None = None, market: dict | None = None, gate_audit: dict | None = None) -> dict:
    row = candidate.copy()
    source_fields = source_fields or {}
    volume_check = volume_check or {}
    portfolio = (market or {}).get("portfolio_mode", {}) or {}
    row.update(source_fields)
    if order:
        row.update({
            "original_candidate_rank": order.get("candidate_rank"),
            "original_account_risk_pct": order.get("account_risk_pct"),
            "mode_account_risk_limit": order.get("mode_account_risk_limit", ""),
            "account_risk_pass": order.get("account_risk_pass", ""),
            "original_estimated_amount": order.get("estimated_amount"),
            "suggested_lots": order.get("suggested_lots", ""),
            "estimated_amount": order.get("estimated_amount", ""),
        })
    row["volume"] = volume_check.get("volume", row.get("volume", ""))
    row["amount"] = volume_check.get("amount", row.get("amount", ""))
    row["review_scope"] = review_scope
    row["original_score"] = candidate.get("score")
    row["original_final_score"] = candidate.get("final_score")
    row["original_reason"] = candidate.get("reason", "")
    row["downgrade_reason"] = "?".join(reason for reason in downgrade_reasons if reason)
    row["reason"] = f"{candidate.get('reason', '')}?{row['downgrade_reason']}"
    row["market_regime"] = row.get("market_regime") or (market or {}).get("market_regime", "")
    row["market_score"] = row.get("market_score") or (market or {}).get("market_score", "")
    row["portfolio_mode"] = row.get("portfolio_mode") or portfolio.get("portfolio_mode", "")
    row["final_portfolio_mode"] = row.get("final_portfolio_mode") or portfolio.get("final_portfolio_mode", portfolio.get("portfolio_mode", ""))
    row["raw_portfolio_mode"] = row.get("raw_portfolio_mode") or portfolio.get("raw_portfolio_mode", portfolio.get("portfolio_mode", ""))
    if gate_audit:
        row.update(gate_audit)
    else:
        row["gate_pass"] = False
        row["gate_failed_reasons"] = list(downgrade_reasons)
        row["gate_checked_fields"] = {}
    return row


def _add_trade_permission_to_rows(rows: list[dict], permission: dict) -> list[dict]:
    for row in rows:
        row["final_trade_permission"] = permission.get("final_trade_permission", "")
        row["final_trade_permission_reason"] = "；".join(str(item) for item in permission.get("final_trade_permission_reason", []) if item)
    return rows


def _finalize_signal_result(result: dict, market: dict, settings: dict) -> dict:
    permission = build_final_trade_permission(market, result, settings)
    result.update(permission)
    result["watchlist"] = _add_trade_permission_to_rows(result.get("watchlist", []), permission)
    result["timing_avoid_list"] = _add_trade_permission_to_rows(result.get("timing_avoid_list", []), permission)
    return result


def build_final_trade_permission(market: dict, signals: dict, settings: dict) -> dict:
    blocker = classify_trading_critical_blockers(settings)
    portfolio = market.get("portfolio_mode", {}) or {}
    if blocker["blocked"]:
        reason = [
            f"blocker_source={blocker['blocker_source']}",
            f"blocker_codes={','.join(blocker['blocker_codes']) or 'none'}",
            f"blocker_severity={blocker['blocker_severity']}",
        ]
        reason.extend(blocker["blocker_reasons"])
        return {"final_trade_permission": "数据不支持正式新开仓", "final_trade_permission_reason": reason}
    if market.get("market_regime") == "cash":
        return {"final_trade_permission": "市场环境禁止新开仓", "final_trade_permission_reason": ["market_regime=cash"]}
    if portfolio.get("portfolio_mode") == "cash":
        return {"final_trade_permission": "组合模式禁止新开仓", "final_trade_permission_reason": ["portfolio_mode=cash"]}
    if not signals.get("candidates"):
        return {"final_trade_permission": "暂无满足数据、时机和风控的正式候选", "final_trade_permission_reason": ["无正式BUY候选"]}
    return {"final_trade_permission": "可按规则轻仓关注正式候选", "final_trade_permission_reason": ["存在通过最终校验的正式候选"]}


def _defensive_probe_note(style: str, portfolio_mode: str) -> str:
    if portfolio_mode == "defensive" and style in TREND_STYLES:
        return "当前为 defensive 模式，该候选虽为趋势类风格，但因账户风险低于 defensive 上限、数据 fresh、仓位小，仅作为小仓位试探，不代表进攻仓。"
    return ""


def _style_state_reason(style: str, state: str) -> str:
    display = STYLE_DISPLAY_NAMES.get(style, style)
    if state == "strong":
        return f"{display} \u98ce\u683c\u5f53\u524d\u8f83\u5f3a\uff0c\u4e2a\u80a1\u6d41\u52a8\u6027\u4e0e\u76f8\u5bf9\u5f3a\u5f31\u8868\u73b0\u8f83\u597d"
    if state == "neutral":
        return f"{display} \u98ce\u683c\u5f53\u524d\u4e2d\u6027\uff0c\u5019\u9009\u80a1\u9700\u901a\u8fc7\u66f4\u4e25\u683c\u7684\u7ec4\u5408\u6a21\u5f0f\u95e8\u69db"
    return f"{display} \u98ce\u683c\u5f53\u524d\u504f\u5f31\uff0c\u4e0d\u5141\u8bb8\u8fdb\u5165\u6b63\u5f0f\u4e70\u5165\u5019\u9009"


def _ranked_candidates(
    stock_frames: dict[str, pd.DataFrame],
    names: dict[str, str],
    filter_reasons: dict[str, list[str]],
    market: dict,
    settings: dict | None = None,
    scores_by_symbol: dict[str, dict] | None = None,
    timing_by_symbol: dict[str, dict] | None = None,
) -> list[dict]:
    passed = []
    settings = settings or {}
    integrity = settings.get("data_integrity_report", {}) or {}
    weights = settings.get("data_integrity", {}) or {}
    signal_weight = float(weights.get("signal_score_weight", 0.70))
    market_weight = float(weights.get("market_strength_weight", 0.20))
    integrity_weight = float(weights.get("score_weight", 0.10))
    market_strength = float(market.get("market_strength", 0) or 0)
    data_integrity_score = float(integrity.get("data_integrity_score", settings.get("data_integrity_score", 1.0)) or 0)
    confidence_multiplier = float(integrity.get("confidence_position_multiplier", settings.get("confidence_position_multiplier", 1.0)) or 1.0)
    style_state_by_style = {row["style"]: row["state"] for row in market.get("style_state_table", [])}
    style_metrics_by_style = {row["style"]: row for row in market.get("style_state_table", [])}
    strongest_styles = {
        style.strip()
        for style in str(market.get("portfolio_mode", {}).get("strongest_styles", "")).split(",")
        if style.strip() and style.strip() not in {"none", "无", "鏃?"}
    }
    for symbol, df in stock_frames.items():
        if filter_reasons.get(symbol):
            continue
        item = (scores_by_symbol or {}).get(symbol)
        if item is None:
            item = score_stock(symbol, names.get(symbol, ""), df)
        else:
            item = item.copy()
        base_signal_score = float(item.get("score", 0) or 0)
        style = item.get("best_style", "other")
        style_metrics = style_metrics_by_style.get(style, {})
        style_momentum_score = float(style_metrics.get("style_momentum_score", base_signal_score) or 0)
        breadth_score = float(style_metrics.get("breadth_score", style_momentum_score) or 0)
        rotation_bonus = float(style_metrics.get("style_rotation_bonus", 0) or 0)
        retreat_penalty = float(style_metrics.get("style_retreat_penalty", 0) or 0)
        signal_score = (
            base_signal_score * 0.50
            + style_momentum_score * 0.25
            + breadth_score * 0.15
            + data_integrity_score * 100.0 * 0.10
            + rotation_bonus
            - retreat_penalty
        )
        signal_score = max(0.0, min(100.0, signal_score))
        final_score = signal_score * signal_weight + market_strength * market_weight + (data_integrity_score * 100.0) * integrity_weight
        item["base_signal_score"] = round(base_signal_score, 2)
        item["signal_score"] = round(signal_score, 2)
        item["style_momentum_score"] = round(style_momentum_score, 2)
        item["breadth_score"] = round(breadth_score, 2)
        item["style_rotation_bonus"] = round(rotation_bonus, 2)
        item["style_retreat_penalty"] = round(retreat_penalty, 2)
        item["retreat_risk"] = bool(style_metrics.get("retreat_risk", False))
        item["market_strength"] = round(market_strength, 2)
        item["data_integrity_score"] = round(data_integrity_score, 4)
        item["confidence_position_multiplier"] = confidence_multiplier
        item["final_score"] = round(final_score, 2)
        _apply_timing_fields(item, (timing_by_symbol or {}).get(symbol))
        if style in style_state_by_style:
            item["style_state"] = style_state_by_style[style]
            item["reason"] = _style_state_reason(style, item["style_state"])
        if item.get("retreat_risk"):
            item["reason"] = f"{item.get('reason', '')}；所属风格出现退潮风险，已扣减信号分"
            item["risk_note"] = f"{item.get('risk_note', '')}；风格扩散度下降或样本转弱，高分个股也不宜追高"
        elif rotation_bonus > 0:
            item["reason"] = f"{item.get('reason', '')}；所属风格出现扩散转强，获得风格启动加分"
        passed.append(item)
    return sorted(
        passed,
        key=lambda item: (
            0 if item.get("best_style") in strongest_styles else 1 if item.get("style_state") == "neutral" else 2,
            -item["final_score"],
        ),
    )


def generate_buy_signals(
    market: dict,
    stock_frames: dict[str, pd.DataFrame],
    names: dict[str, str],
    filter_reasons: dict[str, list[str]],
    settings: dict,
    rules: dict,
    holdings: pd.DataFrame | None = None,
    data_quality_level: str = "fresh",
    fetch_results_by_symbol: dict | None = None,
    scores_by_symbol: dict[str, dict] | None = None,
    timing_by_symbol: dict[str, dict] | None = None,
) -> dict:
    settings = dict(settings)
    settings.setdefault("data_quality_level", data_quality_level)
    forbidden = categorize_forbidden(filter_reasons, names)
    data_issue_list = _data_issue_list(fetch_results_by_symbol, names)
    ranked = _ranked_candidates(stock_frames, names, filter_reasons, market, settings, scores_by_symbol, timing_by_symbol)
    critical_blocker = classify_trading_critical_blockers(settings)
    style_state_by_style = {row["style"]: row["state"] for row in market.get("style_state_table", [])}
    has_style_state_table = "style_state_table" in market
    strongest_styles = {
        style.strip()
        for style in str(market.get("portfolio_mode", {}).get("strongest_styles", "")).split(",")
        if style.strip() and style.strip() not in {"none", "无", "鏃?"}
    }

    portfolio_mode = market.get("portfolio_mode", {}).get("portfolio_mode", "cash")
    market_state = market.get("market_state") or {"green": "bull", "yellow": "neutral", "red": "bear"}.get(market.get("state"), "unknown")
    if market_state in {"bear", "unknown"} and portfolio_mode != "structural_market":
        result = {"candidates": [], "watchlist": [_normalize_observation(item) for item in ranked[:5]], "forbidden": forbidden, "data_issue_list": data_issue_list}
        return _finalize_signal_result(result, market, settings)

    effective_portfolio_mode = portfolio_mode
    market_regime = market.get("market_regime")
    rule = get_position_rule(settings, effective_portfolio_mode)
    max_candidates = int(rule["max_new_candidates"])
    max_total_position = float(rule["max_total_position"])

    total_limit = settings["initial_cash"] * max_total_position
    holding_value = current_holding_market_value(holdings, stock_frames)
    available_new_amount = max(0.0, total_limit - holding_value)
    single_limit = settings["initial_cash"] * float(rule["max_single_position"])
    confidence_multiplier = float((settings.get("data_integrity_report", {}) or {}).get("confidence_position_multiplier", settings.get("confidence_position_multiplier", 1.0)) or 1.0)
    max_single_new_amount = min(single_limit, available_new_amount) * confidence_multiplier
    if available_new_amount <= 0:
        watch = []
        for item in ranked[:8]:
            copied = item.copy()
            copied["reason"] = f"{item['reason']}\uff1b\u65b0\u5f00\u4ed3\u53ef\u7528\u989d\u5ea6\u4e0d\u8db3"
            watch.append(copied)
        result = {"candidates": [], "watchlist": [_normalize_observation(item) for item in watch], "forbidden": forbidden, "style_explanations": [], "data_issue_list": data_issue_list}
        return _finalize_signal_result(result, market, settings)

    selected = []
    watchlist = []
    timing_avoid_list = []
    remaining_amount = available_new_amount
    max_same_style = int(settings.get("max_same_style_candidates", 1))
    style_counts: dict[str, int] = {}
    style_explanations: list[str] = []

    if critical_blocker["blocked"]:
        watch = [
            _normalize_observation({
                **_candidate_review_row(
                    item,
                    review_scope="candidate_data_review",
                    downgrade_reasons=critical_blocker["blocker_reasons"] or critical_blocker["blocker_codes"],
                    market=market,
                ),
                "blocker_codes": critical_blocker["blocker_codes"],
                "blocker_source": critical_blocker["blocker_source"],
                "blocker_severity": critical_blocker["blocker_severity"],
            })
            for item in ranked[:8]
        ]
        result = {"candidates": [], "watchlist": [_normalize_observation(item) for item in watch], "forbidden": forbidden, "style_explanations": [], "data_issue_list": data_issue_list, "timing_avoid_list": []}
        return _finalize_signal_result(result, market, settings)

    for candidate in ranked:
        if len(selected) >= max_candidates:
            watchlist.append(candidate)
            continue
        style = candidate.get("best_style", "other")
        style_state = candidate.get("style_state")
        if style == "other":
            item = candidate.copy()
            item["reason"] = "\u98ce\u683c\u672a\u77e5\uff0c\u9700\u4eba\u5de5\u786e\u8ba4"
            watchlist.append(item)
            continue
        if has_style_state_table and style not in style_state_by_style:
            item = candidate.copy()
            item["reason"] = f"{candidate['reason']}\uff1b\u6240\u5c5e\u98ce\u683c\u672a\u8fdb\u5165\u786e\u8ba4\u98ce\u683c\u6c60"
            watchlist.append(item)
            continue
        if style_state not in {"strong", "neutral"}:
            item = candidate.copy()
            item["reason"] = f"{candidate['reason']}\uff1b\u6240\u5c5e\u98ce\u683c\u5f53\u524d\u504f\u5f31\uff0c\u964d\u7ea7\u89c2\u5bdf"
            watchlist.append(item)
            continue
        minimum_score = candidate_min_score(effective_portfolio_mode, style_state, settings)
        if candidate["final_score"] < minimum_score:
            item = candidate.copy()
            item["minimum_score"] = minimum_score
            item["portfolio_mode"] = portfolio_mode
            item["final_portfolio_mode"] = market.get("portfolio_mode", {}).get("final_portfolio_mode", portfolio_mode)
            item["raw_portfolio_mode"] = market.get("portfolio_mode", {}).get("raw_portfolio_mode", portfolio_mode)
            item["market_regime"] = market_regime or ""
            item["market_score"] = market.get("market_score", "")
            item["review_scope"] = "observation_only"
            item["downgrade_reason"] = f"当前 {portfolio_mode} 模式最低分要求为 {minimum_score:.0f}，该股 final_score={candidate['final_score']:.2f}，分数不足"
            item["reason"] = f"{candidate['reason']}?{item['downgrade_reason']}"
            watchlist.append(item)
            continue
        if not market_regime:
            watchlist.append(
                _candidate_review_row(
                    candidate,
                    review_scope="timing_review",
                    downgrade_reasons=["缺少 market_regime，不能进入正式候选"],
                    market=market,
                )
            )
            continue
        if not candidate.get("timing_decision"):
            watchlist.append(
                _candidate_review_row(
                    candidate,
                    review_scope="timing_review",
                    downgrade_reasons=["缺少 timing_decision，不能进入正式候选"],
                    market=market,
                )
            )
            continue

        candidate = candidate.copy()
        decision = make_final_trade_decision(market_regime, candidate)
        _apply_final_decision_fields(candidate, decision, market)
        if "final_action" not in decision:
            watchlist.append(
                _candidate_review_row(
                    candidate,
                    review_scope="timing_review",
                    downgrade_reasons=["final decision 未返回 final_action，不能进入正式候选"],
                    market=market,
                )
            )
            continue
        if decision.get("final_action") != "BUY":
            item = _candidate_review_row(
                candidate,
                review_scope="timing_review",
                downgrade_reasons=[candidate.get("final_reason") or candidate.get("timing_reason") or "最终交易裁决未达到 BUY"],
                market=market,
            )
            if decision.get("final_action") == "WAIT":
                watchlist.append(item)
            else:
                timing_avoid_list.append(item)
            continue

        if style_counts.get(style, 0) >= max_same_style:
            item = candidate.copy()
            item["reason"] = f"{candidate['reason']}\uff1b\u89e6\u53d1\u540c\u98ce\u683c\u96c6\u4e2d\u5ea6\u9650\u5236\uff0c\u964d\u7ea7\u89c2\u5bdf"
            watchlist.append(item)
            continue

        source_fields = _candidate_source_fields(candidate["symbol"], fetch_results_by_symbol)
        volume_check = _latest_trading_volume_check(candidate["symbol"], stock_frames)
        pre_ok, pre_reasons = validate_pre_order_candidate(candidate, market, source_fields, volume_check, settings)
        if not pre_ok:
            gate_audit = build_candidate_gate_audit(
                candidate, None, market, source_fields, volume_check, settings,
                pre_validation=(pre_ok, pre_reasons),
            )
            watchlist.append(
                _candidate_review_row(
                    candidate,
                    review_scope="candidate_data_review",
                    downgrade_reasons=pre_reasons,
                    source_fields=source_fields,
                    volume_check=volume_check,
                    market=market,
                    gate_audit=gate_audit,
                )
            )
            continue

        market_position_multiplier = float(candidate.get("position_multiplier", 1.0) or 1.0)
        raw_amount = min(max_single_new_amount, remaining_amount) * market_position_multiplier
        order = build_condition_order(candidate, settings, rules, rank=len(selected) + 1, raw_amount=raw_amount)
        mode_account_risk_limit = float(rule.get("max_account_risk_pct", settings.get("account_risk_limit", 0.025)))
        order["mode_account_risk_limit"] = mode_account_risk_limit
        order["account_risk_pass"] = order["account_risk_pct"] <= mode_account_risk_limit
        final_ok, final_reasons = validate_final_candidate(candidate, order, market, source_fields, volume_check, settings)
        if not final_ok:
            gate_audit = build_candidate_gate_audit(
                candidate, order, market, source_fields, volume_check, settings,
                final_validation=(final_ok, final_reasons),
            )
            watchlist.append(
                _candidate_review_row(
                    candidate,
                    review_scope="candidate_risk_review",
                    downgrade_reasons=final_reasons,
                    source_fields=source_fields,
                    volume_check=volume_check,
                    order=order,
                    market=market,
                    gate_audit=gate_audit,
                )
            )
            if style in strongest_styles:
                style_explanations.append(f"{style} 风格存在，但候选未通过账户风险、整手数量或最终校验。")
            continue

        note = _defensive_probe_note(style, portfolio_mode)
        if note:
            order["risk_note"] = f"{order['risk_note']}; {note}"
        order.update(source_fields)
        order["volume"] = volume_check.get("volume", "")
        order["amount"] = volume_check.get("amount", "")
        order.update(
            build_candidate_gate_audit(
                candidate, order, market, source_fields, volume_check, settings,
                final_validation=(final_ok, final_reasons),
            )
        )
        selected.append(order)
        remaining_amount -= order["estimated_amount"]
        style_counts[style] = style_counts.get(style, 0) + 1
        if remaining_amount <= 0:
            break

    if strongest_styles and not selected:
        style_explanations.append(f"strongest_styles: {', '.join(sorted(strongest_styles))}\uff1b\u5f3a\u98ce\u683c\u5b58\u5728\uff0c\u4f46\u6ca1\u6709\u901a\u8fc7\u98ce\u63a7\u3001\u989d\u5ea6\u4e0e\u6570\u636e\u590d\u6838\u7684\u6b63\u5f0f\u4e70\u5165\u5019\u9009\u3002")
    result = {
        "candidates": selected,
        "watchlist": [
            item if item.get("review_scope") in {"candidate_data_review", "candidate_risk_review", "timing_review"} else _normalize_observation(item)
            for item in watchlist[:8]
        ],
        "forbidden": forbidden,
        "style_explanations": style_explanations,
        "data_issue_list": data_issue_list,
        "timing_avoid_list": [_normalize_observation(item) for item in timing_avoid_list[:20]],
    }
    return _finalize_signal_result(result, market, settings)


def generate_buy_signals_from_snapshots(
    market: dict,
    snapshots_by_symbol: dict,
    settings: dict,
    rules: dict,
    holdings: pd.DataFrame | None = None,
    data_quality_level: str = "fresh",
) -> dict:
    scores_by_symbol = {
        symbol: snapshot.score
        for symbol, snapshot in snapshots_by_symbol.items()
        if getattr(snapshot, "score", None) is not None
    }
    timing_by_symbol = {
        symbol: snapshot.timing
        for symbol, snapshot in snapshots_by_symbol.items()
        if getattr(snapshot, "timing", None)
    }
    return generate_buy_signals(
        market,
        snapshots_to_frames(snapshots_by_symbol),
        snapshot_names(snapshots_by_symbol),
        snapshot_filter_reasons(snapshots_by_symbol),
        settings,
        rules,
        holdings,
        data_quality_level,
        snapshot_fetch_results(snapshots_by_symbol),
        scores_by_symbol,
        timing_by_symbol,
    )


def categorize_forbidden(filter_reasons: dict[str, list[str]], names: dict[str, str]) -> dict[str, dict]:
    categories = {
        "overheated": {"rows": [], "extra_count": 0},
        "trend_broken": {"rows": [], "extra_count": 0},
        "abnormal_risk": {"rows": [], "extra_count": 0},
    }
    for symbol, reasons in filter_reasons.items():
        if not reasons:
            continue
        joined = "; ".join(reasons)
        if any(key in joined for key in ["行情数据不可用于交易信号", "daily price data unavailable", "DATA_UNAVAILABLE_FOR_SIGNAL", "no_expected_trade_date_data"]):
            continue
        if any(key in joined for key in ["long upper shadow", "volume spike", "insufficient", "suspended", "ST", "delisting", "failed", "data"]):
            category = "abnormal_risk"
        elif any(key in joined for key in ["MA60", "MA20", "trend"]):
            category = "trend_broken"
        else:
            category = "overheated"
        row = {
            "symbol": symbol,
            "name": names.get(symbol, ""),
            "primary_forbidden_category": category,
            "all_reasons": joined,
            "reasons": joined,
        }
        if len(categories[category]["rows"]) < 20:
            categories[category]["rows"].append(row)
        else:
            categories[category]["extra_count"] += 1
    return categories


def _two_days_below_ma60(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    tail = df.tail(2)
    return bool((tail["close"] < tail["ma60"]).all())


def evaluate_holdings(
    holdings: pd.DataFrame,
    stock_frames: dict[str, pd.DataFrame],
    market: dict,
    settings: dict,
    rules: dict,
) -> list[dict]:
    actions: list[dict] = []
    for _, holding in holdings.iterrows():
        symbol = str(holding["symbol"]).zfill(6)
        name = holding.get("name", "")
        df = stock_frames.get(symbol)
        if df is None or df.empty:
            actions.append({"symbol": symbol, "name": name, "action": "review", "reason": "\u7f3a\u5c11\u6301\u4ed3\u5bf9\u5e94\u884c\u60c5\u6570\u636e\uff0c\u9700\u4eba\u5de5\u590d\u6838"})
            continue
        latest = df.iloc[-1]
        close = float(latest["close"])
        cost = float(holding["cost_price"])
        pnl = (close - cost) / cost if cost else 0.0
        action = "hold"
        reason = "\u8d8b\u52bf\u7ed3\u6784\u5c1a\u672a\u7834\u574f\uff0c\u7ee7\u7eed\u6301\u6709"
        if pnl <= -float(settings["stop_loss_full"]) or _two_days_below_ma60(df) or (market["state"] == "red" and close < latest.get("ma20", 0)):
            action = "exit"
            reason = "\u89e6\u53d1\u5168\u989d\u6b62\u635f\u6216\u8d8b\u52bf\u7834\u574f\uff0c\u5efa\u8bae\u6e05\u4ed3"
        elif pnl <= -float(settings["stop_loss_half"]):
            action = "reduce_half"
            reason = "\u89e6\u53d1\u51cf\u534a\u6b62\u635f\u7ebf\uff0c\u5efa\u8bae\u51cf\u534a"
        elif pnl >= float(settings["take_profit_watch"]) and close < latest.get("ma5", close):
            action = "reduce_half"
            reason = "\u76c8\u5229\u8d85\u8fc7\u89c2\u5bdf\u7ebf\u540e\u8dcc\u7834MA5\uff0c\u5efa\u8bae\u51cf\u534a"
        elif pnl >= float(settings["take_profit_strong"]) and _is_large_bearish_volume(latest):
            action = "reduce_half"
            reason = "\u76c8\u5229\u8d85\u8fc7\u5f3a\u52bf\u6b62\u76c8\u7ebf\u4e14\u51fa\u73b0\u653e\u91cf\u9634\u7ebf\uff0c\u5efa\u8bae\u51cf\u534a"
        actions.append(
            {
                "symbol": symbol,
                "name": name,
                "action": action,
                "reason": reason,
                "close_price": round_price(close),
                "pnl_pct": round(pnl, 4),
            }
        )
    return actions


def evaluate_holdings_from_snapshots(
    holdings: pd.DataFrame,
    snapshots_by_symbol: dict,
    market: dict,
    settings: dict,
    rules: dict,
) -> list[dict]:
    return evaluate_holdings(
        holdings,
        snapshots_to_frames(snapshots_by_symbol),
        market,
        settings,
        rules,
    )


def _is_large_bearish_volume(row: pd.Series) -> bool:
    open_price = float(row.get("open", row["close"]))
    close = float(row["close"])
    amount = float(row.get("amount", 0))
    avg_amount = float(row.get("avg_amount_20d", 0))
    return close < open_price and avg_amount > 0 and amount > avg_amount * 1.5
