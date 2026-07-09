from __future__ import annotations


POSITION_MULTIPLIERS = {
    "attack": 1.0,
    "normal": 0.7,
    "defensive": 0.4,
    "cash": 0.0,
}


def make_final_trade_decision(
    market_regime: str,
    timing_result: dict | None,
    risk_result: dict | None = None,
) -> dict:
    timing_result = timing_result or {}
    timing_decision = timing_result.get("timing_decision")
    reasons: list[str] = []

    if not market_regime:
        return {
            "final_action": "WAIT",
            "final_reason": ["缺少 market_regime，不能进入正式候选"],
            "position_multiplier": 0.0,
            "decision_source": "missing_input",
        }

    if not timing_decision:
        return {
            "final_action": "WAIT",
            "final_reason": ["缺少 timing_decision，不能进入正式候选"],
            "position_multiplier": 0.0,
            "decision_source": "missing_input",
        }

    if market_regime == "cash":
        return {
            "final_action": "AVOID",
            "final_reason": ["市场环境禁止交易"],
            "position_multiplier": 0.0,
            "decision_source": "market_block",
        }

    if timing_decision == "AVOID":
        return {
            "final_action": "AVOID",
            "final_reason": [timing_result.get("timing_reason") or "交易时机为 AVOID"],
            "position_multiplier": 0.0,
            "decision_source": "timing_block",
        }

    if timing_decision == "WAIT":
        return {
            "final_action": "WAIT",
            "final_reason": [timing_result.get("timing_reason") or "交易时机为 WAIT，等待确认"],
            "position_multiplier": 0.0,
            "decision_source": "timing_block",
        }

    if timing_decision != "BUY":
        return {
            "final_action": "WAIT",
            "final_reason": [f"交易时机状态 {timing_decision or '未知'} 未达到 BUY"],
            "position_multiplier": 0.0,
            "decision_source": "timing_block",
        }

    multiplier = POSITION_MULTIPLIERS.get(market_regime, 0.0)
    if market_regime == "attack":
        reasons.append("市场环境为 attack，允许按规则主动交易")
    elif market_regime == "normal":
        reasons.append("市场环境为 normal，允许交易但仓位乘数降为0.7")
    elif market_regime == "defensive":
        reasons.append("市场环境为 defensive，仅允许小仓位防守试探")
    else:
        reasons.append(f"市场环境 {market_regime or '未知'} 不明确，等待确认")
        return {
            "final_action": "WAIT",
            "final_reason": reasons,
            "position_multiplier": 0.0,
            "decision_source": "missing_input",
        }

    if risk_result and risk_result.get("risk_reason"):
        reasons.append(str(risk_result["risk_reason"]))

    return {
        "final_action": "BUY",
        "final_reason": reasons,
        "position_multiplier": multiplier,
        "decision_source": "normal_decision",
    }
