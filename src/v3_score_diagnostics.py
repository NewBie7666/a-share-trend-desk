"""V3 score diagnostics. These helpers never participate in decisions."""

from __future__ import annotations

from math import ceil, floor
from statistics import mean, median


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile
    lower = floor(position)
    upper = ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def score_percentile(value: float, population: list[float]) -> float:
    """Return an inclusive percentile rank in the range 0..100."""
    values = sorted(float(item) for item in population)
    if not values:
        return 0.0
    below = sum(item < float(value) for item in values)
    equal = sum(item == float(value) for item in values)
    rank = (below + 0.5 * equal) / len(values) * 100
    return round(rank, 2)


def analyze_score_distribution(scores: list[int | float]) -> dict:
    values = sorted(float(score) for score in scores)
    if not values:
        return {
            "count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0,
            "p10": 0.0, "p50": 0.0, "p90": 0.0, "score_spread": 0.0,
            "p90_minus_p10": 0.0, "distribution": {},
            "score_concentration_warning": False, "score_dispersion_warning": False,
        }
    buckets = {
        "90_100": sum(90 <= value <= 100 for value in values),
        "80_90": sum(80 <= value < 90 for value in values),
        "70_80": sum(70 <= value < 80 for value in values),
        "60_70": sum(60 <= value < 70 for value in values),
        "below_60": sum(value < 60 for value in values),
    }
    p10 = _percentile(values, 0.10)
    p50 = _percentile(values, 0.50)
    p90 = _percentile(values, 0.90)
    spread = values[-1] - values[0]
    p90_p10 = p90 - p10
    count = len(values)
    return {
        "count": count,
        "min": round(values[0], 2),
        "max": round(values[-1], 2),
        "mean": round(mean(values), 2),
        "median": round(median(values), 2),
        "p10": round(p10, 2),
        "p50": round(p50, 2),
        "p90": round(p90, 2),
        "score_spread": round(spread, 2),
        "p90_minus_p10": round(p90_p10, 2),
        "distribution": {
            key: {"count": value, "ratio": round(value / count, 4)}
            for key, value in buckets.items()
        },
        "score_concentration_warning": buckets["90_100"] / count > 0.20,
        "score_dispersion_warning": count >= 10 and (p90_p10 < 15 or spread < 25),
    }


def build_v3_score_diagnostics(analysis_rows: list[dict], top_rows: list[dict]) -> dict:
    analysis_scores = [float(row.get("stock_factor_score", 0) or 0) for row in analysis_rows]
    top_stock_scores = [float(row.get("stock_factor_score", 0) or 0) for row in top_rows]
    timing_scores = [float(row.get("timing_score", 0) or 0) for row in top_rows]
    total_scores = [float(row.get("total_score", 0) or 0) for row in top_rows]
    analysis = analyze_score_distribution(analysis_scores)
    top_stock = analyze_score_distribution(top_stock_scores)
    timing = analyze_score_distribution(timing_scores)
    total = analyze_score_distribution(total_scores)
    top_stock["score_concentration_warning"] = bool(
        top_stock["score_concentration_warning"] or (top_stock["count"] and top_stock["mean"] > 90)
    )
    return {
        "analysis_pool_stock_factor": analysis,
        "top30_stock_factor": top_stock,
        "top30_timing": timing,
        "top30_total": total,
        "score_concentration_warning": bool(
            analysis["score_concentration_warning"] or top_stock["score_concentration_warning"]
        ),
        "score_dispersion_warning": bool(
            analysis["score_dispersion_warning"] or top_stock["score_dispersion_warning"]
        ),
    }
