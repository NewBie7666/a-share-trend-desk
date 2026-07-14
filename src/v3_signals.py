"""V3 new-position discovery. Existing holding actions are deliberately untouched."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .filters import is_risk_name
from .position_rules import get_position_rule
from .signals import build_condition_order, classify_trading_critical_blockers, current_holding_market_value
from .v3_decision import calculate_timing_score, calculate_v3_score, evaluate_v3_risk
from .v3_market_permission import confirm_v3_market_permission, resolve_v3_market_permission
from .v3_score_diagnostics import build_v3_score_diagnostics, score_percentile


DECISION_STAGES = (
    "universe", "stock_factor", "timing", "market_permission", "style_advisory",
    "risk", "data", "funds_and_account_risk", "final_action",
)


def _decision_step(stage: str, result: str, reason: str, inputs: dict | None = None) -> dict:
    return {"stage": stage, "result": result, "reason": reason, "inputs": inputs or {}}


def _complete_decision_path(row: dict, final_result: str, final_reason: str) -> None:
    existing = {item.get("stage") for item in row.get("decision_path", [])}
    for stage in DECISION_STAGES:
        if stage == "final_action":
            continue
        if stage not in existing:
            row.setdefault("decision_path", []).append(
                _decision_step(stage, "NOT_EVALUATED", "前序条件未通过，本阶段未执行")
            )
    row.setdefault("decision_path", []).append(_decision_step("final_action", final_result, final_reason))
    order = {stage: index for index, stage in enumerate(DECISION_STAGES)}
    row["decision_path"].sort(key=lambda item: order.get(item.get("stage"), len(order)))
    row["decision_path_json"] = json.dumps(row["decision_path"], ensure_ascii=False)


def _market_determinable(market: dict, expected_date: str) -> bool:
    core = {"000300", "000001", "399001"}
    rows = {
        str(item.get("index_code", "")): item
        for item in (market.get("market_index_table", []) or [])
        if str(item.get("index_code", "")) in core
    }
    if set(rows) != core:
        return False
    return all(
        item.get("state") != "failed"
        and (not expected_date or str(item.get("effective_latest_date") or item.get("latest_date") or "") == expected_date)
        for item in rows.values()
    )


def _empty_forbidden() -> dict:
    return {name: {"rows": [], "extra_count": 0} for name in ("overheated", "trend_broken", "abnormal_risk")}


def build_fresh_validation(snapshot, expected_date: str) -> dict:
    result = snapshot.fetch_result
    effective = result.effective_latest_date or result.latest_date or ""
    latest = snapshot.latest
    volume = float(latest.get("volume", 0) or 0) if latest is not None else 0.0
    amount = float(latest.get("amount", 0) or 0) if latest is not None else 0.0
    provider_success = (
        bool(result.provider_success)
        if result.provider_success is not None
        else result.final_source == "fresh"
    )
    if not provider_success:
        reason = "provider_failed" if result.final_source == "failed" else "cache_source"
    elif effective != expected_date or not result.is_expected_trade_date:
        reason = "provider_success_but_stale"
    elif volume <= 0:
        reason = "invalid_volume"
    elif amount <= 0:
        reason = "invalid_amount"
    else:
        reason = "provider_data_valid"
    fresh = reason == "provider_data_valid"
    return {
        "provider_success": provider_success,
        "latest_trade_date": effective,
        "expected_trade_date": expected_date,
        "volume": volume,
        "amount": amount,
        "fresh": fresh,
        "reason": reason,
    }


def _base_row(snapshot, market_score: float, config: dict, filter_rules: dict, expected_date: str) -> dict:
    score = dict(snapshot.score or {})
    latest = snapshot.latest
    timing = snapshot.timing or {}
    structure = str(timing.get("structure_state", ""))
    entry_score = float(timing.get("entry_quality_score", 0) or 0)
    timing_result = calculate_timing_score(entry_score, structure, config)
    risk = evaluate_v3_risk(snapshot.indicators, structure, filter_rules)
    v3_score = calculate_v3_score(float(score.get("score", 0) or 0), market_score, timing_result["timing_score"], risk["risk_adjustment"], config)
    fresh_validation = build_fresh_validation(snapshot, expected_date)
    candidate_source = (
        "fresh"
        if fresh_validation["fresh"]
        else (snapshot.fetch_result.final_source if snapshot.fetch_result.final_source != "fresh" else "invalid")
    )
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
        "candidate_data_source": candidate_source,
        "provider_final_source": snapshot.fetch_result.final_source,
        "candidate_latest_date": snapshot.fetch_result.effective_latest_date or snapshot.fetch_result.latest_date or "",
        "candidate_raw_latest_date": snapshot.fetch_result.raw_latest_date or "",
        "is_expected_trade_date": snapshot.fetch_result.is_expected_trade_date,
        "volume": float(latest.get("volume", 0) or 0) if latest is not None else 0.0,
        "amount": float(latest.get("amount", 0) or 0) if latest is not None else 0.0,
        "fresh_validation": fresh_validation,
        "fresh_validation_json": json.dumps(fresh_validation, ensure_ascii=False),
        "new_position_action": v3_score["v3_action"],
        "minimum_stock_factor_for_buy": float(config.get("minimum_stock_factor_for_buy", 60)),
        "holding_action": "UNCHANGED",
        "holding_risk_note": "V3仅评估新增开仓，不改变已有持仓动作",
        "decision_path": [],
        "future_check": {},
        "style_is_advisory": True,
        "style_not_gate": True,
        "v3_style_role": "advisory_only",
        "risk_components_json": json.dumps(risk.get("risk_components", []), ensure_ascii=False),
        "risk_adjustment_audit_json": json.dumps(risk.get("risk_adjustment_audit", {}), ensure_ascii=False),
        "timing_reason_codes_json": json.dumps(timing.get("timing_reason_codes", ["TIMING_REASON_UNAVAILABLE"]), ensure_ascii=False),
        "timing_reason_text": "；".join(timing.get("timing_reason_texts", [])) or "时机原因不可用",
    }
    return row


def _intrinsic_risk_reasons(snapshot, row: dict) -> list[str]:
    reasons: list[str] = []
    if is_risk_name(snapshot.name):
        reasons.append("ST、退市整理或退市风险")
    if row.get("structure_state") == "breakdown":
        reasons.append("结构明确破位，仅禁止新增开仓")
    if row.get("extreme_risk"):
        reasons.append(f"复合极端风险：{row.get('risk_reason', '')}")
    return reasons


def _ranking_action(row: dict) -> str:
    if row.get("v3_action") == "BUY":
        return "QUALIFIED"
    if row.get("v3_action") == "WATCH":
        return "WATCH"
    return "LOW_SCORE"


def _fresh_reason_text(validation: dict) -> str:
    mapping = {
        "provider_data_valid": "Provider数据fresh、交易日及成交有效",
        "provider_success_but_stale": "Provider请求成功但行情日期不是预期交易日",
        "provider_failed": "Provider请求失败或行情不可用",
        "cache_source": "仅有缓存数据，正式新开仓要求Provider fresh数据",
        "invalid_volume": "预期交易日成交量无效",
        "invalid_amount": "预期交易日成交额无效",
    }
    return mapping.get(str(validation.get("reason", "")), "数据fresh校验未通过")


def generate_v3_signals_from_snapshots(market: dict, snapshots_by_symbol: dict, settings: dict, rules: dict, holdings: pd.DataFrame | None = None) -> dict:
    config = settings.get("v3", {}) or {}
    expected_date = str(settings.get("expected_trade_date", ""))
    market_score = float(market.get("market_score", 0) or 0)
    blocker = classify_trading_critical_blockers(settings)
    raw_permission = resolve_v3_market_permission(
        market_score,
        critical_blocked=bool(blocker["blocked"]),
        market_determinable=_market_determinable(market, expected_date),
    )
    permission_cfg = config.get("market_permission", {}) or {}
    state_path_value = settings.get("v3_market_permission_state_path")
    permission = confirm_v3_market_permission(
        raw_permission,
        expected_date or str(settings.get("report_date", "")),
        int(permission_cfg.get("confirmation_days", 2)),
        Path(state_path_value) if state_path_value else None,
    )
    multiplier = float(permission["v3_position_multiplier"])
    filter_rules = rules.get("filters", rules)
    analysis_rows = [
        {
            "symbol": snapshot.symbol,
            "stock_factor_score": float((snapshot.score or {}).get("score", 0) or 0),
        }
        for snapshot in snapshots_by_symbol.values()
        if snapshot.score is not None
    ]
    scored_rows: list[dict] = []
    for snapshot in snapshots_by_symbol.values():
        if not snapshot.timing or snapshot.score is None:
            continue
        row = _base_row(snapshot, market_score, config, filter_rules, expected_date)
        row.update(permission)
        row["ranking_action"] = _ranking_action(row)
        row["score_action"] = row.get("v3_action", "")
        row["position_multiplier"] = multiplier
        row["market_multiplier"] = multiplier
        intrinsic_reasons = _intrinsic_risk_reasons(snapshot, row)
        row["intrinsic_risk_reasons"] = intrinsic_reasons
        fresh_reason = _fresh_reason_text(row["fresh_validation"])
        row["decision_path"].append(_decision_step("universe", "PASS", "进入V3评分池", {"symbol": row["symbol"]}))
        row["decision_path"].append(_decision_step(
            "stock_factor", "PASS" if row["ranking_action"] == "QUALIFIED" else "WATCH",
            row["v3_reason"], {"stock_factor_score": row["stock_factor_score"], "total_score": row["total_score"]},
        ))
        row["decision_path"].append(_decision_step("timing", "PASS", f"timing_score={row['timing_score']:.2f}", {"structure_state": row.get("structure_state", "")}))
        row["decision_path"].append(_decision_step(
            "market_permission", "PASS" if permission["v3_market_permission"] != "BLOCKED" else "BLOCK",
            "；".join(permission["v3_market_reason"]),
            {"market_score": market_score, "permission": permission["v3_market_permission"], "multiplier": multiplier},
        ))
        row["decision_path"].append(_decision_step("style_advisory", "PASS", "风格仅用于排序解释，不构成V3硬门控", {"best_style": row.get("best_style", "other")}))
        row["decision_path"].append(_decision_step("risk", "BLOCK" if intrinsic_reasons else "PASS", "；".join(intrinsic_reasons) or row["risk_reason"], {"risk_flags": row.get("risk_flags", [])}))
        row["decision_path"].append(_decision_step("data", "PASS" if row["fresh_validation"]["fresh"] else "BLOCK", fresh_reason, row["fresh_validation"]))

        if not row["fresh_validation"]["fresh"]:
            row.update(eligibility_status="DATA_BLOCKED", new_position_action="BLOCKED", v3_action="BLOCKED", v3_reason=fresh_reason)
        elif intrinsic_reasons:
            reason = "；".join(intrinsic_reasons)
            row.update(eligibility_status="RISK_BLOCKED", new_position_action="BLOCKED", v3_action="BLOCKED", v3_reason=reason)
        elif row["ranking_action"] != "QUALIFIED":
            row.update(eligibility_status="WATCH_ONLY", new_position_action="WATCH", v3_action="WATCH")
        elif permission["v3_market_permission"] == "BLOCKED":
            reason = "；".join(permission["v3_market_reason"])
            row.update(eligibility_status="MARKET_BLOCKED", new_position_action="BLOCKED", v3_action="BLOCKED", v3_reason=reason, market_block_reason=reason)
        else:
            row.update(eligibility_status="BUY_QUALIFIED", new_position_action="BUY", market_block_reason="")
        scored_rows.append(row)

    scored_rows.sort(key=lambda item: (-float(item.get("total_score", 0)), item["symbol"]))
    ranked = scored_rows[: int(config.get("top_pool_size", 30))]
    analysis_scores = [float(row["stock_factor_score"]) for row in analysis_rows]
    timing_scores = [float(row.get("timing_score", 0) or 0) for row in ranked]
    total_scores = [float(row.get("total_score", 0) or 0) for row in ranked]
    for rank, row in enumerate(ranked, start=1):
        row["v3_rank"] = rank
        row["score_percentile"] = score_percentile(float(row.get("stock_factor_score", 0) or 0), analysis_scores)
        row["timing_score_percentile"] = score_percentile(float(row.get("timing_score", 0) or 0), timing_scores)
        row["total_score_percentile"] = score_percentile(float(row.get("total_score", 0) or 0), total_scores)
    score_diagnostics = build_v3_score_diagnostics(analysis_rows, ranked)

    frames = {symbol: snapshot.indicators for symbol, snapshot in snapshots_by_symbol.items() if snapshot.usable_for_signal}
    holding_value = current_holding_market_value(holdings, frames)
    base_rule = get_position_rule(settings, permission["v3_position_mode"])
    initial_cash = float(settings.get("initial_cash", 30000))
    base_total_amount = initial_cash * float(base_rule["max_total_position"])
    max_total_amount = base_total_amount * multiplier
    available_amount = max(0.0, max_total_amount - holding_value)
    base_single_amount = initial_cash * float(base_rule["max_single_position"])
    max_single_amount = base_single_amount * multiplier
    account_risk_limit = float(base_rule.get("max_account_risk_pct", settings.get("account_risk_limit", 0.025)))
    max_candidates = min(int(config.get("max_candidates", 3)), int(permission["v3_max_candidates"]))

    candidates: list[dict] = []
    watchlist: list[dict] = []
    blocked: list[dict] = []
    remaining = available_amount
    for row in ranked:
        if row["eligibility_status"] in {"DATA_BLOCKED", "RISK_BLOCKED", "MARKET_BLOCKED"}:
            _complete_decision_path(row, "BLOCK", row["v3_reason"])
            blocked.append(row)
            continue
        if row["eligibility_status"] == "WATCH_ONLY":
            _complete_decision_path(row, "WATCH", row["v3_reason"])
            watchlist.append(row)
            continue
        if multiplier <= 0 or remaining <= 0 or len(candidates) >= max_candidates:
            row.update(eligibility_status="RISK_BLOCKED", new_position_action="BLOCKED", v3_action="BLOCKED")
            row["v3_reason"] = "新增仓位余额不足或正式候选数量已达上限"
            _complete_decision_path(row, "BLOCK", row["v3_reason"])
            blocked.append(row)
            continue

        slots = max(max_candidates - len(candidates), 1)
        base_position_amount = min(base_single_amount, base_total_amount / slots)
        market_limited_amount = min(max_single_amount, remaining / slots, remaining)
        row["final_score"] = row["total_score"]
        row["score"] = row["stock_factor_score"]
        row.setdefault("best_style", "other")
        row.setdefault("style_state", "")
        row.setdefault("style_scores", {})
        row.setdefault("reason", row["v3_reason"])
        row.setdefault("risk_note", row["risk_reason"])
        order = build_condition_order(row, settings, rules, len(candidates) + 1, market_limited_amount)
        risk_limited_amount = market_limited_amount
        if float(order["account_risk_pct"]) > account_risk_limit:
            trigger = float(order.get("trigger_price", 0) or 0)
            final_exit = float(order.get("final_exit_price", 0) or 0)
            loss_fraction = (trigger - final_exit) / trigger if trigger > 0 else 0
            if loss_fraction > 0:
                risk_limited_amount = min(market_limited_amount, initial_cash * account_risk_limit / loss_fraction)
                order = build_condition_order(row, settings, rules, len(candidates) + 1, risk_limited_amount)
        risk_pass = float(order["account_risk_pct"]) <= account_risk_limit
        lots_pass = int(order["suggested_lots"]) >= 1
        order.update(row)
        order.update({
            "mode_account_risk_limit": account_risk_limit,
            "account_risk_pass": risk_pass,
            "base_position_amount": round(base_position_amount, 2),
            "market_multiplier": multiplier,
            "risk_limited_amount": round(risk_limited_amount, 2),
            "final_order_amount": round(float(order.get("estimated_amount", 0) or 0), 2),
        })
        order["decision_path"].append(_decision_step(
            "funds_and_account_risk", "PASS" if risk_pass and lots_pass else "BLOCK",
            f"整手={order['suggested_lots']}，账户风险通过={risk_pass}",
            {"risk_limit": account_risk_limit, "final_order_amount": order["final_order_amount"]},
        ))
        if not risk_pass or not lots_pass:
            order.update(eligibility_status="RISK_BLOCKED", new_position_action="BLOCKED", v3_action="BLOCKED")
            order["v3_reason"] = "账户风险未通过" if not risk_pass else "资金不足一手"
            row.update(
                eligibility_status="RISK_BLOCKED",
                new_position_action="BLOCKED",
                v3_action="BLOCKED",
                v3_reason=order["v3_reason"],
                account_risk_pass=risk_pass,
                suggested_lots=order.get("suggested_lots", 0),
                final_order_amount=order.get("final_order_amount", 0),
            )
            _complete_decision_path(order, "BLOCK", order["v3_reason"])
            blocked.append(order)
            continue
        order.update(eligibility_status="BUY_QUALIFIED", new_position_action="BUY", v3_action="BUY")
        row.update(
            eligibility_status="BUY_QUALIFIED",
            new_position_action="BUY",
            v3_action="BUY",
            account_risk_pass=True,
            suggested_lots=order.get("suggested_lots", 0),
            final_order_amount=order.get("final_order_amount", 0),
        )
        _complete_decision_path(order, "PASS", "通过V3新增开仓全部校验")
        candidates.append(order)
        remaining -= float(order["estimated_amount"])

    for row in ranked:
        row.setdefault("decision_path_json", json.dumps(row.get("decision_path", []), ensure_ascii=False))
        row.setdefault("review_scope", "v3_scoring")
        row.setdefault("downgrade_reason", row.get("v3_reason", ""))

    status_counts = {
        status: sum(1 for row in ranked if row.get("eligibility_status") == status)
        for status in ("BUY_QUALIFIED", "WATCH_ONLY", "MARKET_BLOCKED", "DATA_BLOCKED", "RISK_BLOCKED")
    }
    no_buy_reasons = []
    if not candidates:
        labels = {
            "WATCH_ONLY": "股票评分或Timing/综合分不足",
            "MARKET_BLOCKED": "市场权限禁止",
            "DATA_BLOCKED": "数据非fresh或成交无效",
            "RISK_BLOCKED": "个股、账户、资金或整手风险",
        }
        no_buy_reasons = [f"{labels[key]}：{count}只" for key, count in status_counts.items() if count and key in labels]

    qualified = [row for row in ranked if row.get("ranking_action") == "QUALIFIED"]
    market_pass = qualified if permission["v3_market_permission"] != "BLOCKED" else []
    fresh_pass = [row for row in market_pass if row.get("fresh_validation", {}).get("fresh")]
    intrinsic_pass = [row for row in fresh_pass if not row.get("intrinsic_risk_reasons")]
    scan_count = int(settings.get("pool_available", settings.get("scan_count", len(snapshots_by_symbol))) or 0)
    signal_funnel = {
        "scan": {"input_count": scan_count, "output_count": len(snapshots_by_symbol), "reject_count": max(scan_count - len(snapshots_by_symbol), 0), "reject_reason": "未进入分析池"},
        "base_filter": {"input_count": len(snapshots_by_symbol), "output_count": len(scored_rows), "reject_count": max(len(snapshots_by_symbol) - len(scored_rows), 0), "reject_reason": "缺少评分或Timing结果"},
        "top30": {"input_count": len(scored_rows), "output_count": len(ranked), "reject_count": max(len(scored_rows) - len(ranked), 0), "reject_reason": "综合排名低于Top30"},
        "scored": {"input_count": len(ranked), "output_count": len(qualified), "reject_count": len(ranked) - len(qualified), "reject_reason": "股票因子分或综合分不足"},
        "market_permission_check": {"input_count": len(qualified), "output_count": len(market_pass), "reject_count": len(qualified) - len(market_pass), "reject_reason": "市场权限禁止新增开仓"},
        "fresh_check": {"input_count": len(market_pass), "output_count": len(fresh_pass), "reject_count": len(market_pass) - len(fresh_pass), "reject_reason": "数据非fresh、日期或成交无效"},
        "risk_check": {"input_count": len(fresh_pass), "output_count": len(intrinsic_pass), "reject_count": len(fresh_pass) - len(intrinsic_pass), "reject_reason": "个股新增开仓硬风险"},
        "final_buy": {"input_count": len(intrinsic_pass), "output_count": len(candidates), "reject_count": max(len(intrinsic_pass) - len(candidates), 0), "reject_reason": "账户风险、资金、整手或候选上限"},
    }
    return {
        "decision_engine": "v3", **permission,
        "candidates": candidates,
        "watchlist": watchlist[:30],
        "blocked_list": blocked,
        "timing_avoid_list": [],
        "forbidden": _empty_forbidden(),
        "data_issue_list": [],
        "ranked": ranked,
        "score_diagnostics": score_diagnostics,
        "signal_funnel": signal_funnel,
        "v3_decision_funnel": {**status_counts, "top30": len(ranked), "final_buy": len(candidates)},
        "why_no_buy": no_buy_reasons,
        "final_trade_permission": "V3市场权限允许按规则关注正式候选" if candidates else ("V3市场权限禁止正式新开仓" if permission["v3_market_permission"] == "BLOCKED" else "暂无满足V3评分、数据与风控的正式候选"),
        "final_trade_permission_reason": permission["v3_market_reason"] + no_buy_reasons,
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
