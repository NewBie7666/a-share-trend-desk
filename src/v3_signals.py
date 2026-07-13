"""V3 new-position discovery. Existing holding actions are deliberately untouched."""

from __future__ import annotations

import json

import pandas as pd

from .filters import is_risk_name
from .position_rules import get_position_rule
from .signals import build_condition_order, current_holding_market_value
from .v3_decision import calculate_timing_score, calculate_v3_score, evaluate_v3_risk, market_position_multiplier


def _decision_step(stage: str, result: str, reason: str) -> dict:
    return {"stage": stage, "result": result, "reason": reason}


def _empty_forbidden() -> dict:
    return {name: {"rows": [], "extra_count": 0} for name in ("overheated", "trend_broken", "abnormal_risk")}


def _source_ok(snapshot, expected_date: str) -> tuple[bool, str]:
    result = snapshot.fetch_result
    effective = result.effective_latest_date or result.latest_date or ""
    if result.final_source != "fresh":
        return False, "正式新开仓要求 fresh 数据"
    if effective != expected_date or not result.is_expected_trade_date:
        return False, "有效行情日期不是预期交易日"
    return True, "fresh 且交易日匹配"


def _base_row(snapshot, market_score: float, config: dict, filter_rules: dict) -> dict:
    score = dict(snapshot.score or {})
    latest = snapshot.latest
    timing = snapshot.timing or {}
    structure = str(timing.get("structure_state", ""))
    entry_score = float(timing.get("entry_quality_score", 0) or 0)
    timing_result = calculate_timing_score(entry_score, structure, config)
    risk = evaluate_v3_risk(snapshot.indicators, structure, filter_rules)
    v3_score = calculate_v3_score(float(score.get("score", 0) or 0), market_score, timing_result["timing_score"], risk["risk_adjustment"], config)
    row = {
        **score,
        **timing,
        **timing_result,
        **risk,
        **v3_score,
        "symbol": snapshot.symbol,
        "name": snapshot.name,
        "signal_price": round(float(latest.get("close", 0) or 0), 2) if latest is not None else 0.0,
        "close_price": round(float(latest.get("close", 0) or 0), 2) if latest is not None else 0.0,
        "ma20": float(latest.get("ma20", 0) or 0) if latest is not None else 0.0,
        "ma60": float(latest.get("ma60", 0) or 0) if latest is not None else 0.0,
        "candidate_data_source": snapshot.fetch_result.final_source,
        "candidate_latest_date": snapshot.fetch_result.effective_latest_date or snapshot.fetch_result.latest_date or "",
        "candidate_raw_latest_date": snapshot.fetch_result.raw_latest_date or "",
        "is_expected_trade_date": snapshot.fetch_result.is_expected_trade_date,
        "volume": float(latest.get("volume", 0) or 0) if latest is not None else 0.0,
        "amount": float(latest.get("amount", 0) or 0) if latest is not None else 0.0,
        "new_position_action": v3_score["v3_action"],
        "minimum_stock_factor_for_buy": float(config.get("minimum_stock_factor_for_buy", 60)),
        "holding_action": "UNCHANGED",
        "holding_risk_note": "V3仅评估新增开仓，不改变已有持仓动作",
        "decision_path": [],
        "future_check": {},
    }
    return row


def _hard_block_reasons(snapshot, row: dict, expected_date: str) -> list[str]:
    reasons: list[str] = []
    if snapshot.data_issue is not None or not snapshot.usable_for_signal:
        reasons.append("行情完全失败、历史不足或无有效交易日数据")
    if is_risk_name(snapshot.name):
        reasons.append("ST、退市整理或退市风险")
    if row.get("volume", 0) <= 0 or row.get("amount", 0) <= 0:
        reasons.append("停牌或预期交易日成交量/成交额为0")
    if row.get("structure_state") == "breakdown":
        reasons.append("结构明确破位，仅禁止新增开仓")
    if row.get("extreme_risk"):
        reasons.append(f"复合极端风险：{row.get('risk_reason', '')}")
    return reasons


def generate_v3_signals_from_snapshots(market: dict, snapshots_by_symbol: dict, settings: dict, rules: dict, holdings: pd.DataFrame | None = None) -> dict:
    config = settings.get("v3", {}) or {}
    expected_date = str(settings.get("expected_trade_date", ""))
    market_score = float(market.get("market_score", 0) or 0)
    multiplier = market_position_multiplier(market_score)
    filter_rules = rules.get("filters", rules)
    ranked = []
    for snapshot in snapshots_by_symbol.values():
        if not snapshot.timing or snapshot.score is None:
            continue
        row = _base_row(snapshot, market_score, config, filter_rules)
        hard_reasons = _hard_block_reasons(snapshot, row, expected_date)
        row["position_multiplier"] = multiplier
        row["decision_path"].append(_decision_step("stock_factor", "PASS", f"stock_factor_score={row['stock_factor_score']:.2f}"))
        row["decision_path"].append(_decision_step("timing", "PASS", f"timing_score={row['timing_score']:.2f}"))
        row["decision_path"].append(_decision_step("market", "PASS" if multiplier > 0 else "BLOCK", f"market_score={market_score:.2f}, multiplier={multiplier:.2f}"))
        row["decision_path"].append(_decision_step("risk", "BLOCK" if hard_reasons else "PASS", "；".join(hard_reasons) or row["risk_reason"]))
        row["decision_path"].append(_decision_step("v3_score", row["v3_action"], f"total_score={row['total_score']:.2f}"))
        if hard_reasons:
            row["v3_action"] = "BLOCKED"
            row["new_position_action"] = "BLOCKED"
            row["v3_reason"] = "；".join(hard_reasons)
        ranked.append(row)
    ranked.sort(key=lambda item: (-float(item.get("total_score", 0)), item["symbol"]))
    top_limit = int(config.get("top_pool_size", 30))
    ranked = ranked[:top_limit]
    for rank, row in enumerate(ranked, start=1):
        row["v3_rank"] = rank

    frames = {symbol: snapshot.indicators for symbol, snapshot in snapshots_by_symbol.items() if snapshot.usable_for_signal}
    holding_value = current_holding_market_value(holdings, frames)
    base_rule = get_position_rule(settings, "attack")
    initial_cash = float(settings.get("initial_cash", 30000))
    max_total_amount = initial_cash * float(base_rule["max_total_position"]) * multiplier
    available_amount = max(0.0, max_total_amount - holding_value)
    max_single_amount = initial_cash * float(base_rule["max_single_position"]) * multiplier
    account_risk_limit = float(settings.get("account_risk_limit", 0.025))
    max_candidates = int(config.get("max_candidates", 3))

    candidates: list[dict] = []
    watchlist: list[dict] = []
    ignored: list[dict] = []
    blocked: list[dict] = []
    remaining = available_amount
    for row in ranked:
        if row["v3_action"] == "BLOCKED":
            blocked.append(row)
            continue
        if row["v3_action"] != "BUY":
            (watchlist if row["v3_action"] == "WATCH" else ignored).append(row)
            continue
        source_ok, source_reason = _source_ok(snapshots_by_symbol[row["symbol"]], expected_date)
        row["decision_path"].append(_decision_step("data", "PASS" if source_ok else "BLOCK", source_reason))
        if not source_ok:
            row["new_position_action"] = "WATCH"
            row["v3_action"] = "WATCH"
            row["v3_reason"] = source_reason
            watchlist.append(row)
            continue
        if multiplier <= 0 or remaining <= 0 or len(candidates) >= max_candidates:
            row["new_position_action"] = "WATCH"
            row["v3_action"] = "WATCH"
            row["v3_reason"] = "市场仓位乘数或新增仓位余额不足"
            watchlist.append(row)
            continue
        slots = max(max_candidates - len(candidates), 1)
        raw_amount = min(max_single_amount, remaining / slots, remaining)
        row["final_score"] = row["total_score"]
        row["score"] = row["stock_factor_score"]
        row.setdefault("best_style", "other")
        row.setdefault("style_state", "")
        row.setdefault("style_scores", {})
        row.setdefault("reason", row["v3_reason"])
        row.setdefault("risk_note", row["risk_reason"])
        order = build_condition_order(row, settings, rules, len(candidates) + 1, raw_amount)
        if float(order["account_risk_pct"]) > account_risk_limit:
            trigger = float(order.get("trigger_price", 0) or 0)
            final_exit = float(order.get("final_exit_price", 0) or 0)
            loss_fraction = (trigger - final_exit) / trigger if trigger > 0 else 0
            if loss_fraction > 0:
                risk_capped_amount = initial_cash * account_risk_limit / loss_fraction
                order = build_condition_order(
                    row, settings, rules, len(candidates) + 1, min(raw_amount, risk_capped_amount)
                )
        risk_pass = float(order["account_risk_pct"]) <= account_risk_limit
        lots_pass = int(order["suggested_lots"]) >= 1
        order.update(row)
        order["mode_account_risk_limit"] = account_risk_limit
        order["account_risk_pass"] = risk_pass
        order["new_position_action"] = "BUY" if risk_pass and lots_pass else "WATCH"
        order["decision_path"].append(_decision_step("funds_and_risk", "PASS" if risk_pass and lots_pass else "BLOCK", f"lots={order['suggested_lots']}, account_risk_pass={risk_pass}"))
        order["decision_path_json"] = json.dumps(order["decision_path"], ensure_ascii=False)
        if not risk_pass or not lots_pass:
            order["v3_action"] = "WATCH"
            order["v3_reason"] = "账户风险未通过" if not risk_pass else "资金不足一手"
            watchlist.append(order)
            continue
        candidates.append(order)
        remaining -= float(order["estimated_amount"])

    for row in ranked:
        row.setdefault("decision_path_json", json.dumps(row.get("decision_path", []), ensure_ascii=False))
        row.setdefault("review_scope", "v3_scoring")
        row.setdefault("downgrade_reason", row.get("v3_reason", ""))

    no_buy_reasons = []
    if not candidates:
        for label, rows in (("硬禁止", blocked), ("观察", watchlist), ("评分不足", ignored)):
            if rows:
                no_buy_reasons.append(f"{label}：{len(rows)}只")
    return {
        "decision_engine": "v3",
        "candidates": candidates,
        "watchlist": watchlist[:20] + ignored[:10],
        "blocked_list": blocked,
        "timing_avoid_list": [],
        "forbidden": _empty_forbidden(),
        "data_issue_list": [],
        "ranked": ranked,
        "v3_decision_funnel": {
            "top30": len(ranked), "blocked": len(blocked), "watch": len(watchlist),
            "ignored": len(ignored), "final_buy": len(candidates),
        },
        "why_no_buy": no_buy_reasons,
        "final_trade_permission": "可按规则关注V3正式候选" if candidates else "暂无满足V3评分、数据与风控的正式候选",
        "final_trade_permission_reason": no_buy_reasons,
    }


def annotate_v3_holding_boundary(holding_actions: list[dict], snapshots_by_symbol: dict, settings: dict, rules: dict) -> list[dict]:
    config = settings.get("v3", {}) or {}
    filter_rules = rules.get("filters", rules)
    output = []
    for item in holding_actions:
        row = item.copy()
        symbol = str(row.get("symbol", "")).zfill(6)
        snapshot = snapshots_by_symbol.get(symbol)
        extreme = False
        risk_reason = ""
        if snapshot is not None and snapshot.timing and snapshot.usable_for_signal:
            structure = str(snapshot.timing.get("structure_state", ""))
            risk = evaluate_v3_risk(snapshot.indicators, structure, filter_rules)
            extreme = bool(risk["extreme_risk"])
            risk_reason = risk["risk_reason"]
        row["existing_holding_action"] = row.get("action", "")
        row["holding_action"] = "UNCHANGED"
        row["new_position_action"] = "BLOCKED" if extreme else "NOT_APPLICABLE"
        row["holding_risk_note"] = (
            f"{risk_reason}；该风险仅限制新增开仓，不构成已有持仓卖出信号"
            if extreme else "V3不改变已有持仓动作"
        )
        output.append(row)
    return output
