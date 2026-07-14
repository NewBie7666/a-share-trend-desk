"""V3 market permission resolution and confirmation state."""

from __future__ import annotations

import json
from pathlib import Path

from .utils import CACHE_DIR, ensure_directories


PERMISSION_ORDER = {
    "BLOCKED": 0,
    "LIMITED": 1,
    "DEFENSIVE": 2,
    "BALANCED": 3,
    "ATTACK": 4,
}


def resolve_v3_market_permission(
    market_score: float,
    critical_blocked: bool,
    market_determinable: bool,
) -> dict:
    """Resolve the raw V3 permission without reading V2 portfolio mode."""
    score = float(market_score or 0)
    if critical_blocked:
        values = ("BLOCKED", "cash", 0.0, 0, ["关键数据阻断，禁止V3正式新开仓"])
    elif not market_determinable:
        values = ("BLOCKED", "cash", 0.0, 0, ["核心市场状态不可判定，禁止V3正式新开仓"])
    elif score < 20:
        values = ("BLOCKED", "cash", 0.0, 0, [f"market_score={score:.2f}，低于20分"])
    elif score < 40:
        values = ("LIMITED", "defensive", 0.25, 1, [f"market_score={score:.2f}，仅允许受限仓位"])
    elif score < 60:
        values = ("DEFENSIVE", "defensive", 0.50, 2, [f"market_score={score:.2f}，允许防守仓位"])
    elif score < 80:
        values = ("BALANCED", "balanced", 0.80, 2, [f"market_score={score:.2f}，允许均衡仓位"])
    else:
        values = ("ATTACK", "attack", 1.00, 3, [f"market_score={score:.2f}，允许进攻仓位"])
    permission, mode, multiplier, candidates, reasons = values
    return {
        "raw_v3_market_permission": permission,
        "v3_market_permission": permission,
        "permission_raw_target": permission,
        "permission_final": permission,
        "permission_confirmation_required": permission in {"BALANCED", "ATTACK"},
        "permission_confirmation_status": "not_applied",
        "permission_adjustment_reason": "",
        "v3_position_mode": mode,
        "v3_position_multiplier": multiplier,
        "v3_max_candidates": candidates,
        "v3_market_reason": reasons,
    }


def _permission_values(permission: str) -> tuple[str, float, int]:
    mapping = {
        "BLOCKED": ("cash", 0.0, 0),
        "LIMITED": ("defensive", 0.25, 1),
        "DEFENSIVE": ("defensive", 0.50, 2),
        "BALANCED": ("balanced", 0.80, 2),
        "ATTACK": ("attack", 1.00, 3),
    }
    return mapping.get(permission, mapping["BLOCKED"])


def confirm_v3_market_permission(
    raw_result: dict,
    trade_date: str,
    confirmation_days: int = 2,
    state_path: Path | None = None,
) -> dict:
    """Confirm only BALANCED/ATTACK upgrades; baseline permissions apply now."""
    ensure_directories()
    path = state_path or (CACHE_DIR / "v3_market_permission_state.json")
    raw_permission = str(raw_result.get("raw_v3_market_permission", "BLOCKED"))
    confirmation_days = max(int(confirmation_days or 1), 1)
    state_error = ""
    state_exists = path.exists()
    try:
        state = json.loads(path.read_text(encoding="utf-8")) if state_exists else {}
        if state and str(state.get("confirmed_permission", "")) not in PERMISSION_ORDER:
            raise ValueError("invalid confirmed_permission")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        state = {}
        state_error = f"权限确认缓存不可用，已保守重建：{exc}"

    confirmed = str(state.get("confirmed_permission") or "BLOCKED")
    pending = str(state.get("pending_permission") or "")
    count = int(state.get("pending_confirmation_days", 0) or 0)
    first_date = str(state.get("pending_first_date") or "")
    last_date = str(state.get("last_trade_date") or "")
    raw_rank = PERMISSION_ORDER.get(raw_permission, 0)
    confirmed_rank = PERMISSION_ORDER.get(confirmed, 0)

    confirmation_required = raw_permission in {"BALANCED", "ATTACK"}
    confirmation_status = "not_required"
    adjustment_reason = ""

    if raw_permission == "BLOCKED":
        confirmed = "BLOCKED"
        pending, count, first_date = "", 0, ""
        confirmation_status = "blocked"
    elif raw_permission in {"LIMITED", "DEFENSIVE"}:
        # These are baseline permissions determined directly by market_score.
        # A stale/corrupt historical BLOCKED state must never weaken them.
        confirmed = raw_permission
        pending, count, first_date = "", 0, ""
        confirmation_status = "not_required"
    elif raw_rank <= confirmed_rank:
        confirmed = raw_permission
        pending, count, first_date = "", 0, ""
        confirmation_status = "downgrade_applied" if raw_rank < confirmed_rank else "confirmed"
    else:
        if pending != raw_permission:
            pending, count, first_date = raw_permission, 1, trade_date
        elif trade_date and trade_date != last_date:
            count += 1
        if count >= confirmation_days:
            confirmed = raw_permission
            pending, count, first_date = "", 0, ""
            confirmation_status = "confirmed"
        else:
            confirmed = "DEFENSIVE" if raw_permission == "BALANCED" else "BALANCED"
            confirmation_status = "pending"
            adjustment_reason = (
                f"{raw_permission}升级需连续{confirmation_days}个不同交易日确认，"
                f"当前{count}/{confirmation_days}，暂按{confirmed}执行"
            )

    mode, multiplier, max_candidates = _permission_values(confirmed)
    reason = list(raw_result.get("v3_market_reason", []))
    if adjustment_reason:
        reason.append(adjustment_reason)
    if state_error:
        reason.append(state_error)
    payload = {
        "confirmed_permission": confirmed,
        "pending_permission": pending,
        "pending_confirmation_days": count,
        "pending_first_date": first_date,
        "last_trade_date": trade_date,
        "raw_permission": raw_permission,
    }
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)
    return {
        **raw_result,
        "v3_market_permission": confirmed,
        "permission_raw_target": raw_permission,
        "permission_final": confirmed,
        "permission_confirmation_required": confirmation_required,
        "permission_confirmation_status": confirmation_status,
        "permission_adjustment_reason": adjustment_reason,
        "v3_position_mode": mode,
        "v3_position_multiplier": multiplier,
        "v3_max_candidates": max_candidates,
        "v3_market_reason": reason,
        "market_permission_confirmation_days": confirmation_days,
        "market_permission_pending": pending,
        "market_permission_confirmed_days": count,
        "market_permission_state_path": str(path),
    }
