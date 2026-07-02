from __future__ import annotations


DEFAULT_POSITION_RULES = {
    "attack": {
        "max_total_position": 0.80,
        "max_single_position": 0.30,
        "max_new_candidates": 3,
        "min_score": 70,
        "max_account_risk_pct": 0.030,
    },
    "balanced": {
        "max_total_position": 0.50,
        "max_single_position": 0.25,
        "max_new_candidates": 2,
        "min_score": 75,
        "max_account_risk_pct": 0.025,
    },
    "defensive": {
        "max_total_position": 0.30,
        "max_single_position": 0.20,
        "max_new_candidates": 1,
        "min_score": 78,
        "max_account_risk_pct": 0.015,
    },
    "cash": {
        "max_total_position": 0.00,
        "max_single_position": 0.00,
        "max_new_candidates": 0,
        "min_score": 999,
        "max_account_risk_pct": 0.000,
    },
}


def get_position_rule(settings: dict, portfolio_mode: str) -> dict:
    configured = settings.get("position_rules", {}) or {}
    rule = {**DEFAULT_POSITION_RULES.get(portfolio_mode, DEFAULT_POSITION_RULES["cash"])}
    rule.update(configured.get(portfolio_mode, {}) or {})
    if "max_account_risk_pct" not in rule:
        rule["max_account_risk_pct"] = DEFAULT_POSITION_RULES.get(portfolio_mode, DEFAULT_POSITION_RULES["cash"])["max_account_risk_pct"]
    return rule


def trade_permission_text(portfolio_mode: str) -> str:
    return {
        "attack": "可进攻交易",
        "balanced": "轻仓试探，不追高",
        "defensive": "防守仓位，仅允许低账户风险、低金额、fresh 数据候选；不追高，不加码。",
        "cash": "不新开仓",
    }.get(portfolio_mode, "不新开仓")


def operation_summary(portfolio_mode: str, strongest_styles: str = "无", potential_strong_styles: str = "无") -> str:
    base = {
        "attack": "今日为 attack 模式，可按规则关注正式候选，但仍需控制单票风险。",
        "balanced": "今日为 balanced 模式，仅允许轻仓试探；不追高，不补位。",
        "defensive": "今日为 defensive 模式，采用防守仓位；仅允许低账户风险、低金额、fresh 数据候选，不追高，不加码。",
        "cash": "今日为 cash 模式，不新开仓，只处理已有持仓。",
    }.get(portfolio_mode, "今日不新开仓，只处理已有持仓。")
    if strongest_styles == "无" and potential_strong_styles and potential_strong_styles != "无":
        base += " 当前无高置信度强风格，潜在强风格样本不足，不支持进攻仓位。"
    return base


def candidate_min_score(portfolio_mode: str, style_state: str, settings: dict) -> float:
    rule = get_position_rule(settings, portfolio_mode)
    minimum = float(rule["min_score"])
    if portfolio_mode == "balanced" and style_state == "neutral":
        return max(minimum, 82.0)
    return minimum
