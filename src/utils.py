from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover - fallback for minimal bundled runtimes
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
REPORT_DIR = DATA_DIR / "reports"
LOG_DIR = DATA_DIR / "logs"
HOLDINGS_PATH = DATA_DIR / "holdings.csv"

HOLDINGS_COLUMNS = [
    "symbol",
    "name",
    "quantity",
    "cost_price",
    "buy_date",
    "stop_loss_price",
    "take_profit_price",
    "note",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        if yaml is not None:
            return yaml.safe_load(fh) or {}
        return _simple_yaml_load(fh.read())


def _parse_scalar(value: str) -> Any:
    value = value.strip().strip('"').strip("'")
    if value == "":
        return ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Small fallback parser for this project's simple config files."""
    lines = [line.rstrip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if any(line.strip().startswith("- ") for line in lines):
        return _simple_yaml_load_list_file(lines)
    return _simple_yaml_load_nested_map(lines)


def _simple_yaml_load_list_file(lines: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current_key: str | None = None
    current_item: dict[str, Any] | None = None
    for line in lines:
        stripped = line.strip()
        if not line.startswith(" ") and stripped.endswith(":"):
            current_key = stripped[:-1]
            result[current_key] = []
            continue
        if not line.startswith(" ") and ":" in stripped:
            key, value = stripped.split(":", 1)
            result[key] = _parse_scalar(value)
            current_key = None
            continue
        if current_key and stripped.startswith("- "):
            content = stripped[2:]
            if ":" in content:
                key, value = content.split(":", 1)
                current_item = {key.strip(): _parse_scalar(value)}
                result[current_key].append(current_item)
            else:
                result[current_key].append(_parse_scalar(content))
            continue
        if current_key and current_item is not None and ":" in stripped:
            key, value = stripped.split(":", 1)
            current_item[key.strip()] = _parse_scalar(value)
            continue
    return result


def _simple_yaml_load_nested_map(lines: list[str]) -> dict[str, Any]:
    nested_result: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, nested_result)]
    for line in lines:
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            parent[key] = {}
            stack.append((indent, parent[key]))
        else:
            parent[key] = _parse_scalar(value)
    return nested_result


def load_config() -> dict[str, Any]:
    settings = load_yaml(CONFIG_DIR / "settings.yaml")
    _apply_runtime_profile(settings)
    return {
        "settings": settings,
        "stock_pool": load_yaml(CONFIG_DIR / "stock_pool.yaml").get("stocks", []),
        "risk_rules": load_yaml(CONFIG_DIR / "risk_rules.yaml"),
    }


def _apply_runtime_profile(settings: dict[str, Any]) -> None:
    profile_name = settings.get("runtime_profile", "fast")
    profiles = settings.get("profiles", {}) or {}
    profile = profiles.get(profile_name, {}) or {}
    if "max_scan_symbols" in profile:
        settings["max_scan_symbols"] = profile["max_scan_symbols"]
    profile_fetch = profile.get("data_fetch", {}) or {}
    if profile_fetch:
        fetch_cfg = dict(settings.get("data_fetch", {}) or {})
        fetch_cfg.update(profile_fetch)
        settings["data_fetch"] = fetch_cfg


def ensure_directories() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def ensure_holdings_file(path: Path = HOLDINGS_PATH) -> Path:
    ensure_directories()
    if not path.exists():
        pd.DataFrame(columns=HOLDINGS_COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def load_holdings(path: Path = HOLDINGS_PATH) -> pd.DataFrame:
    ensure_holdings_file(path)
    df = pd.read_csv(path, dtype={"symbol": str})
    for col in HOLDINGS_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[HOLDINGS_COLUMNS]


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def round_price(value: float) -> float:
    return round(float(value), 2)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"
