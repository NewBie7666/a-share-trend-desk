from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import random
import time

import pandas as pd

from .utils import CACHE_DIR, LOG_DIR, ensure_directories, ensure_holdings_file


def _install_requests_default_timeout(timeout_seconds: float = 10.0) -> None:
    """Apply a default timeout to third-party requests calls that omit one."""
    import requests

    session_cls = requests.sessions.Session
    current = session_cls.request
    if getattr(current, "_a_share_timeout_wrapper", False):
        return

    original = current

    def request_with_timeout(self, method, url, **kwargs):
        kwargs.setdefault("timeout", timeout_seconds)
        return original(self, method, url, **kwargs)

    request_with_timeout._a_share_timeout_wrapper = True
    session_cls.request = request_with_timeout


@dataclass
class FetchResult:
    symbol: str
    name: str
    data: pd.DataFrame
    used_cache: bool
    error: str | None = None
    final_source: str = "fresh"
    latest_date: str | None = None
    raw_latest_date: str | None = None
    effective_latest_date: str | None = None
    trimmed_future_rows_count: int = 0
    expected_trade_date: str | None = None
    is_expected_trade_date: bool = True
    data_staleness_reason: str = ""
    error_category: str = ""
    attempt_logs: list[dict] | None = None
    success_attempt: int | None = None
    attempt_count: int = 0
    last_error: str = ""
    fallback_reason: str = ""
    data_fetch_abort_reason: str = ""
    fetch_started_at: str = ""
    fetch_finished_at: str = ""
    data_source_name: str = ""
    fallback_source_used: str = ""
    source_attempts: list[dict] | None = None


def is_main_board_symbol(symbol: str) -> bool:
    symbol = str(symbol).zfill(6)
    return symbol.startswith(("600", "601", "603", "605", "000", "001", "002"))


def is_risk_stock_name(name: str) -> bool:
    text = str(name).upper()
    return "ST" in text or "退" in str(name)


def _normalize_spot_columns(df: pd.DataFrame) -> pd.DataFrame:
    column_map = {
        "代码": "symbol",
        "名称": "name",
        "最新价": "price",
        "成交额": "amount",
        "涨跌幅": "pct_change",
        "所属行业": "industry",
        "行业": "industry",
    }
    out = df.rename(columns=column_map).copy()
    for col in ["symbol", "name", "price", "amount", "industry"]:
        if col not in out.columns:
            out[col] = "未知行业" if col == "industry" else pd.NA
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    return out.dropna(subset=["symbol", "name"])


def _normalize_index_constituents(df: pd.DataFrame) -> set[str]:
    for column in ["品种代码", "证券代码", "代码", "con_code", "symbol"]:
        if column in df.columns:
            return {
                str(value).split(".")[0].zfill(6)
                for value in df[column].dropna().tolist()
                if str(value).strip()
            }
    return set()


def _index_constituent_symbols(ak, index_code: str) -> set[str]:
    try:
        return _normalize_index_constituents(ak.index_stock_cons(symbol=index_code))
    except Exception:
        return set()


def _pool_metadata(pool: list[dict], source: str, version: str) -> dict:
    distribution: dict[str, int] = {}
    for item in pool:
        industry = str(item.get("industry") or "未知行业")
        distribution[industry] = distribution.get(industry, 0) + 1
    return {
        "pool_version": version,
        "pool_size": len(pool),
        "pool_source": source,
        "industry_distribution": dict(sorted(distribution.items())),
    }


def _spot_provider_result(ak, source: str, expected_date: str) -> tuple[pd.DataFrame, dict]:
    started = time.perf_counter()
    try:
        raw = ak.stock_zh_a_spot_em() if source == "akshare_eastmoney_spot" else ak.stock_zh_a_spot()
        return _normalize_spot_columns(raw), {
            "source": source,
            "status": "success",
            "latency": round(time.perf_counter() - started, 3),
            "symbol_count": len(raw),
            "raw_count": len(raw),
            "symbol_valid_count": 0,
            "main_board_count": 0,
            "st_filter_count": 0,
            "liquidity_filter_count": 0,
            "final_count": 0,
            "pass_ratio": 0.0,
            "schema_quality": 1.0,
            "data_quality": "fresh",
            "latest_trade_date": expected_date,
            "data_role": "stock_basic",
        }
    except Exception as exc:  # pragma: no cover - depends on remote provider
        return pd.DataFrame(), {
            "source": source,
            "status": "failed",
            "latency": round(time.perf_counter() - started, 3),
            "symbol_count": 0,
            "raw_count": 0,
            "symbol_valid_count": 0,
            "main_board_count": 0,
            "st_filter_count": 0,
            "liquidity_filter_count": 0,
            "final_count": 0,
            "pass_ratio": 0.0,
            "schema_quality": 0.0,
            "data_quality": "failed",
            "latest_trade_date": "",
            "data_role": "stock_basic",
            "error": str(exc),
        }


def load_main_board_stock_pool(settings: dict, fallback_pool: list[dict[str, str]]) -> tuple[list[dict[str, str]], bool, str | None, dict]:
    version = str(settings.get("pool_version", "v2.1"))
    expected_date = str(settings.get("expected_trade_date", ""))
    if settings.get("stock_pool_mode", "manual") != "main_board_all":
        return fallback_pool, False, None, _pool_metadata(fallback_pool, "本地白名单", version)
    try:
        if settings.get("akshare_runtime_unavailable"):
            raise ConnectionError("AkShare startup preflight timed out; use local stock-pool cache")
        import akshare as ak
        _install_requests_default_timeout(float((settings.get("data_fetch", {}) or {}).get("network_timeout_seconds", 10)))
        providers: list[dict] = []
        spot = pd.DataFrame()
        for source in ["akshare_eastmoney_spot", "akshare_legacy_spot"]:
            spot, provider = _spot_provider_result(ak, source, expected_date)
            providers.append(provider)
            if not spot.empty:
                break
        if spot.empty:
            raise ConnectionError("; ".join(item.get("error", item["source"]) for item in providers))
        lot_size = int(settings.get("lot_size", 100))
        max_one_lot = float(settings.get("prefilter_max_one_lot_amount", settings.get("initial_cash", 30000)))
        min_amount = float(settings.get("prefilter_min_amount", settings.get("min_avg_amount_20d", 300000000)))
        main_board = spot[spot["symbol"].map(is_main_board_symbol)].copy()
        after_st = main_board[~main_board["name"].map(is_risk_stock_name)].copy()
        after_liquidity = after_st[after_st["amount"] >= min_amount].copy()
        filtered = after_liquidity[(after_liquidity["price"] > 0) & (after_liquidity["price"] * lot_size <= max_one_lot)].copy()
        filtered = filtered.sort_values("amount", ascending=False).drop_duplicates("symbol")
        provider = providers[-1]
        provider.update({
            "symbol_valid_count": len(spot),
            "main_board_count": len(main_board),
            "st_filter_count": len(after_st),
            "liquidity_filter_count": len(after_liquidity),
            "final_count": len(filtered),
            "pass_ratio": round(len(filtered) / max(int(provider["raw_count"]), 1), 4),
        })
        hs300_symbols = _index_constituent_symbols(ak, "000300")
        csi500_symbols = _index_constituent_symbols(ak, "000905")
        indexed_symbols = hs300_symbols | csi500_symbols
        max_scan = int(settings.get("max_scan_symbols", 500))

        leaders = pd.DataFrame(columns=filtered.columns)
        if "industry" in filtered.columns:
            known_industry = filtered[filtered["industry"].notna() & (filtered["industry"] != "未知行业")]
            if not known_industry.empty:
                leaders = known_industry.groupby("industry", group_keys=False).head(1)
        indexed = filtered[filtered["symbol"].isin(indexed_symbols)]
        ordered = pd.concat([leaders, indexed, filtered], ignore_index=True).drop_duplicates("symbol").head(max_scan)
        pool = []
        for row in ordered.itertuples(index=False):
            sources = []
            if row.symbol in hs300_symbols:
                sources.append("沪深300")
            if row.symbol in csi500_symbols:
                sources.append("中证500")
            if str(getattr(row, "industry", "未知行业")) != "未知行业" and any(row.symbol == leader.symbol for leader in leaders.itertuples(index=False)):
                sources.append("行业龙头")
            if not sources:
                sources.append("成交额筛选")
            pool.append(
                {
                    "symbol": row.symbol,
                    "name": row.name,
                    "industry": str(getattr(row, "industry", "未知行业") or "未知行业"),
                    "price": float(getattr(row, "price", 0) or 0),
                    "amount": float(getattr(row, "amount", 0) or 0),
                    "turnover": float(getattr(row, "turnover", 0) or 0),
                    "pool_source": "+".join(sources),
                }
            )
        if not pool:
            raise ValueError("full market prefilter returned empty pool")
        provider["final_count"] = len(pool)
        _write_stock_pool_cache(pool, settings, "实时provider")
        metadata = _pool_metadata(pool, "沪深300+中证500+主板行业龙头+成交额筛选", version)
        metadata.update(
            {
                "providers": providers,
                "pool_funnel": {
                    "raw_count": int(providers[-1]["symbol_count"]),
                    "main_board_count": len(main_board),
                    "after_ST_filter": len(after_st),
                    "after_liquidity_filter": len(after_liquidity),
                    "after_data_filter": len(filtered),
                    "final_scan_count": len(pool),
                },
                "pool_components": {
                    "hs300": len(set(item["symbol"] for item in pool) & hs300_symbols),
                    "csi500": len(set(item["symbol"] for item in pool) & csi500_symbols),
                    "industry_leader": len(leaders),
                    "liquidity_fill": len(pool),
                },
            }
        )
        return pool, False, None, metadata
    except Exception as exc:  # pragma: no cover - network fallback depends on environment
        cached, cache_meta = _read_stock_pool_cache(settings)
        if cached:
            metadata = _pool_metadata(cached, "本地缓存股票池", version)
            metadata.update({
                "providers": [{"source": "local_cache", "status": "fallback", "latency": 0.0, "symbol_count": len(cached), "latest_trade_date": cache_meta.get("trade_date", ""), "data_role": "stock_basic", "error": str(exc)}],
                "pool_funnel": {"raw_count": len(cached), "main_board_count": len(cached), "after_ST_filter": len(cached), "after_liquidity_filter": len(cached), "after_data_filter": len(cached), "final_scan_count": len(cached)},
                "pool_components": {"local_cache": len(cached)},
                "cache_metadata": cache_meta,
            })
            return cached, True, str(exc), metadata
        metadata = _pool_metadata(fallback_pool, "本地备用白名单", version)
        metadata.update({
            "providers": [{"source": "local_whitelist", "status": "fallback", "latency": 0.0, "symbol_count": len(fallback_pool), "latest_trade_date": "", "data_role": "stock_basic", "error": str(exc)}],
            "pool_funnel": {"raw_count": len(fallback_pool), "main_board_count": len(fallback_pool), "after_ST_filter": len(fallback_pool), "after_liquidity_filter": len(fallback_pool), "after_data_filter": len(fallback_pool), "final_scan_count": len(fallback_pool)},
            "pool_components": {"local_whitelist": len(fallback_pool)},
            "cache_metadata": cache_meta,
        })
        return fallback_pool, True, str(exc), metadata


def _stock_pool_cache_path() -> Path:
    return CACHE_DIR / "main_board_pool_cache.json"


def _legacy_stock_pool_cache_path() -> Path:
    return CACHE_DIR / "main_board_pool.csv"


def _write_stock_pool_cache(pool: list[dict[str, str]], settings: dict | None = None, source: str = "") -> None:
    ensure_directories()
    cfg = (settings or {}).get("pool_cache", {}) or {}
    payload = {
        "cache_version": int(cfg.get("cache_version", 1)),
        "filter_version": str(cfg.get("filter_version", "v2.1-pr3")),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "trade_date": str((settings or {}).get("expected_trade_date", "")),
        "symbol_count": len(pool),
        "records": pool,
    }
    _stock_pool_cache_path().write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_stock_pool_cache(settings: dict | None = None) -> tuple[list[dict], dict]:
    path = _stock_pool_cache_path()
    cfg = (settings or {}).get("pool_cache", {}) or {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            compatible = payload.get("cache_version") == int(cfg.get("cache_version", 1)) and payload.get("filter_version") == str(cfg.get("filter_version", "v2.1-pr3"))
            created = datetime.fromisoformat(str(payload.get("created_at", "")))
            age = (datetime.now() - created).days
            if compatible:
                return list(payload.get("records", [])), {"cache_age_days": age, "cache_compatible": True, **payload}
            return [], {"cache_age_days": age, "cache_compatible": False, "reject_reason": "缓存版本或筛选版本不匹配"}
        except (OSError, ValueError, json.JSONDecodeError):
            return [], {"cache_compatible": False, "reject_reason": "缓存JSON不可读取"}
    legacy = _legacy_stock_pool_cache_path()
    if not legacy.exists():
        return [], {"cache_compatible": False, "reject_reason": "缓存不存在"}
    df = pd.read_csv(legacy, dtype={"symbol": str})
    if df.empty:
        return [], {"cache_compatible": False, "reject_reason": "旧缓存为空"}
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    records = [
        {
            "symbol": row.symbol,
            "name": row.name,
            "industry": str(getattr(row, "industry", "未知行业") or "未知行业"),
            "pool_source": str(getattr(row, "pool_source", "本地缓存") or "本地缓存"),
        }
        for row in df.itertuples(index=False)
    ]
    return records, {"cache_compatible": False, "reject_reason": "仅发现旧CSV缓存，需实时provider刷新后迁移"}


def _normalize_daily_columns(df: pd.DataFrame) -> pd.DataFrame:
    column_map = {
        "\u65e5\u671f": "date",
        "\u5f00\u76d8": "open",
        "\u6700\u9ad8": "high",
        "\u6700\u4f4e": "low",
        "\u6536\u76d8": "close",
        "\u6210\u4ea4\u91cf": "volume",
        "\u6210\u4ea4\u989d": "amount",
        "\u6da8\u8dcc\u5e45": "pct_change",
        "\u6362\u624b\u7387": "turnover",
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "amount": "amount",
    }
    out = df.rename(columns=column_map).copy()
    required = ["date", "open", "high", "low", "close", "volume", "amount"]
    for col in required:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[required + [c for c in ["pct_change", "turnover"] if c in out.columns]]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["open", "close", "high", "low", "volume", "amount"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


def _ak_stock_symbol(symbol: str) -> str:
    symbol = str(symbol).zfill(6)
    return ("sh" if symbol.startswith(("6", "9")) else "sz") + symbol


def _index_symbol_without_prefix(ak_symbol: str, index_code: str) -> str:
    return index_code or str(ak_symbol).replace("sh", "").replace("sz", "")


def _source_attempt(
    *,
    source_name: str,
    status: str,
    final_source: str = "",
    error: str = "",
    error_category: str = "",
    raw_latest_date: str | None = None,
    effective_latest_date: str | None = None,
    trimmed_future_rows_count: int = 0,
    rows: int = 0,
) -> dict:
    return {
        "source_name": source_name,
        "status": status,
        "final_source": final_source,
        "error": error,
        "error_category": error_category,
        "raw_latest_date": raw_latest_date or "",
        "effective_latest_date": effective_latest_date or "",
        "trimmed_future_rows_count": trimmed_future_rows_count,
        "rows": rows,
    }


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}_daily.csv"


def _index_cache_path() -> Path:
    return CACHE_DIR / "hs300_daily.csv"


def _generic_index_cache_path(index_code: str) -> Path:
    return CACHE_DIR / f"index_{index_code}_daily.csv"


def read_cache(symbol: str) -> pd.DataFrame:
    path = _cache_path(symbol)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _write_cache(symbol: str, df: pd.DataFrame) -> None:
    ensure_directories()
    df.to_csv(_cache_path(symbol), index=False, encoding="utf-8-sig")


def latest_date_of(df: pd.DataFrame) -> str | None:
    if df.empty or "date" not in df.columns:
        return None
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.max().strftime("%Y-%m-%d")


def _trim_to_expected_trade_date(df: pd.DataFrame, expected_trade_date_value: str | None) -> tuple[pd.DataFrame, str | None, str | None, int]:
    raw_latest_date = latest_date_of(df)
    if df.empty or not expected_trade_date_value or "date" not in df.columns:
        return df.copy(), raw_latest_date, raw_latest_date, 0
    dates = pd.to_datetime(df["date"], errors="coerce")
    expected = pd.to_datetime(expected_trade_date_value)
    future_mask = dates > expected
    trimmed_count = int(future_mask.sum())
    trimmed = df.loc[~future_mask].copy().reset_index(drop=True)
    effective_latest_date = latest_date_of(trimmed)
    return trimmed, raw_latest_date, effective_latest_date, trimmed_count


def categorize_error(exc: Exception | str) -> str:
    text = str(exc)
    lowered = text.lower()
    if "no_expected_trade_date_data" in lowered or "no expected trade date data" in lowered:
        return "no_expected_trade_date_data"
    if "data_not_ready" in lowered or "data not ready" in lowered:
        return "data_not_ready"
    if "insufficient_history" in lowered or "insufficient history" in lowered:
        return "insufficient_history"
    if "empty" in lowered or "空" in text:
        return "empty_data"
    if "missing" in lowered or "schema" in lowered or "缺少" in text:
        return "invalid_schema"
    if "stale_cache" in lowered:
        return "stale_cache"
    if "stale" in lowered or "不是预期交易日" in text:
        return "stale_data"
    if "cache missing" in lowered:
        return "cache_missing"
    if any(key in lowered for key in ["remote", "connection", "timeout", "ssl", "http", "network", "aborted"]):
        return "network_error"
    return "unknown"


def _is_network_error(exc: Exception | str) -> bool:
    text = str(exc).lower()
    markers = [
        "remotedisconnected",
        "connection aborted",
        "readtimeout",
        "connectionreseterror",
        "temporary failure",
        "remote end closed connection",
        "connection reset",
        "timeout",
    ]
    return any(marker in text for marker in markers) or categorize_error(exc) == "network_error"


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _sleep_with_jitter(base_seconds: float, jitter_seconds: float, sleep_fn=time.sleep) -> float:
    jitter = random.uniform(0, max(0.0, float(jitter_seconds))) if jitter_seconds else 0.0
    wait = float(base_seconds) + jitter
    sleep_fn(wait)
    return wait


def _validate_daily(df: pd.DataFrame, data_type: str = "daily") -> None:
    if df.empty:
        raise ValueError(f"akshare returned empty {data_type} data")
    required = {"date", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required daily columns: {sorted(missing)}")


def _weekday_fallback(now: datetime) -> str:
    day = now.date()
    cutoff = datetime.strptime("15:10", "%H:%M").time()
    if now.weekday() < 5 and now.time() >= cutoff:
        return day.strftime("%Y-%m-%d")
    day = day - timedelta(days=1)
    while day.weekday() >= 5:
        day = day - timedelta(days=1)
    return day.strftime("%Y-%m-%d")


def get_trade_calendar_dates(settings: dict | None = None) -> tuple[list[str], str]:
    settings = settings if settings is not None else {}
    cache_path = CACHE_DIR / "trade_calendar.json"
    try:
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            dates = payload.get("trade_dates", payload) if isinstance(payload, dict) else payload
            dates = sorted(str(item) for item in dates if item)
            if dates:
                settings["trade_calendar_error"] = ""
                settings["trade_calendar_source"] = "local_cache"
                return dates, "fresh"
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        settings["trade_calendar_error"] = f"本地交易日历缓存不可用：{exc}"
    settings.setdefault("trade_calendar_error", "本地交易日历缓存不存在，使用工作日规则推断")
    settings["trade_calendar_source"] = "fallback_weekday"
    return [], "fallback_weekday"


def expected_trade_date(now: datetime | None = None, trade_dates: list[str] | None = None, settings: dict | None = None) -> tuple[str, str, list[str]]:
    now = now or datetime.now()
    if trade_dates is None:
        trade_dates, status = get_trade_calendar_dates(settings)
    else:
        status = "fresh"
    if not trade_dates:
        return _weekday_fallback(now), "fallback_weekday", []
    today = now.strftime("%Y-%m-%d")
    cutoff = datetime.strptime("15:10", "%H:%M").time()
    eligible = [d for d in trade_dates if d < today or (d == today and now.time() >= cutoff)]
    if not eligible:
        return _weekday_fallback(now), "fallback_weekday", trade_dates
    return eligible[-1], status, trade_dates


def trade_day_diff(start_date: str | None, end_date: str | None, trade_dates: list[str] | None = None) -> int | None:
    if not start_date or not end_date:
        return None
    if start_date > end_date:
        return None
    if trade_dates:
        return sum(1 for d in trade_dates if start_date < d <= end_date)
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    return len(pd.bdate_range(start + timedelta(days=1), end))


def _make_result(
    symbol: str,
    name: str,
    df: pd.DataFrame,
    *,
    final_source: str,
    expected_trade_date_value: str | None,
    error: str | None = None,
    error_category: str = "",
    attempt_logs: list[dict] | None = None,
    success_attempt: int | None = None,
    attempt_count: int = 0,
    last_error: str = "",
    fallback_reason: str = "",
    fetch_started_at: str = "",
    fetch_finished_at: str = "",
    data_source_name: str = "",
    fallback_source_used: str = "",
    source_attempts: list[dict] | None = None,
    min_history_days: int = 0,
) -> FetchResult:
    trimmed_df, raw_latest_date, effective_latest_date, trimmed_count = _trim_to_expected_trade_date(df, expected_trade_date_value)
    is_expected = True if expected_trade_date_value is None else effective_latest_date == expected_trade_date_value
    source = final_source
    category = error_category
    reason = ""
    if expected_trade_date_value is not None and not is_expected:
        source = "failed"
        category = "no_expected_trade_date_data" if effective_latest_date is None else "data_not_ready"
        reason = "data does not contain expected trade-date daily bar" if effective_latest_date is None else "data source has not updated to expected trade date"
    if source != "failed" and min_history_days and len(trimmed_df) < min_history_days:
        source = "failed"
        category = "insufficient_history"
        reason = f"insufficient history after trim: {len(trimmed_df)} < {min_history_days}"
    return FetchResult(
        symbol=symbol,
        name=name,
        data=trimmed_df,
        used_cache=source == "cache",
        error=error,
        final_source=source,
        latest_date=effective_latest_date,
        raw_latest_date=raw_latest_date,
        effective_latest_date=effective_latest_date,
        trimmed_future_rows_count=trimmed_count,
        expected_trade_date=expected_trade_date_value,
        is_expected_trade_date=is_expected,
        data_staleness_reason=reason,
        error_category=category,
        attempt_logs=attempt_logs or [],
        success_attempt=success_attempt,
        attempt_count=attempt_count,
        last_error=last_error,
        fallback_reason=fallback_reason,
        fetch_started_at=fetch_started_at,
        fetch_finished_at=fetch_finished_at,
        data_source_name=data_source_name,
        fallback_source_used=fallback_source_used,
        source_attempts=source_attempts or [],
    )


def _empty_result(
    symbol: str,
    name: str,
    *,
    error: str | None,
    expected_trade_date_value: str | None,
    error_category: str,
    attempt_logs: list[dict] | None = None,
    attempt_count: int = 0,
    last_error: str = "",
    fallback_reason: str = "",
    data_fetch_abort_reason: str = "",
    fetch_started_at: str = "",
    fetch_finished_at: str = "",
    data_source_name: str = "",
    fallback_source_used: str = "",
    source_attempts: list[dict] | None = None,
) -> FetchResult:
    return FetchResult(
        symbol=symbol,
        name=name,
        data=pd.DataFrame(),
        used_cache=False,
        error=error,
        final_source="failed",
        expected_trade_date=expected_trade_date_value,
        is_expected_trade_date=False,
        data_staleness_reason=error or "",
        error_category=error_category,
        attempt_logs=attempt_logs or [],
        attempt_count=attempt_count,
        last_error=last_error or error or "",
        fallback_reason=fallback_reason,
        data_fetch_abort_reason=data_fetch_abort_reason,
        fetch_started_at=fetch_started_at,
        fetch_finished_at=fetch_finished_at,
        data_source_name=data_source_name,
        fallback_source_used=fallback_source_used,
        source_attempts=source_attempts or [],
    )


def _read_valid_cache(
    symbol: str,
    name: str,
    *,
    expected_trade_date_value: str | None,
    max_cache_age_trade_days: int,
    trade_dates: list[str] | None,
    last_error: str,
    attempt_logs: list[dict],
    attempt_count: int = 0,
    fetch_started_at: str = "",
    fetch_finished_at: str = "",
    cache_reader=read_cache,
    min_history_days: int = 0,
    source_attempts: list[dict] | None = None,
) -> FetchResult:
    source_attempts = source_attempts or []
    cached = cache_reader(symbol)
    fallback_reason = "all_retries_failed_network_error" if categorize_error(last_error) == "network_error" else f"all_retries_failed_{categorize_error(last_error)}"
    if cached.empty:
        last_category = categorize_error(last_error)
        if last_category in {"data_not_ready", "insufficient_history", "no_expected_trade_date_data"}:
            attempt_logs.append({"attempt": 0, "error": "cache missing", "error_category": "cache_missing", "final_source": "failed", "attempt_count": attempt_count, "last_error": last_error, "fallback_reason": "cache_missing"})
            result = _empty_result(symbol, name, error=last_error, expected_trade_date_value=expected_trade_date_value, error_category=last_category, attempt_logs=attempt_logs, attempt_count=attempt_count, last_error=last_error, fallback_reason="cache_missing", fetch_started_at=fetch_started_at, fetch_finished_at=fetch_finished_at, data_source_name="local_cache", source_attempts=source_attempts)
            if source_attempts:
                result.raw_latest_date = source_attempts[-1].get("raw_latest_date") or None
                result.effective_latest_date = source_attempts[-1].get("effective_latest_date") or None
                result.latest_date = result.effective_latest_date
                result.trimmed_future_rows_count = int(source_attempts[-1].get("trimmed_future_rows_count") or 0)
            return result
        attempt_logs.append({"attempt": 0, "error": "cache missing", "error_category": "cache_missing", "final_source": "failed", "attempt_count": attempt_count, "last_error": last_error, "fallback_reason": "cache_missing"})
        return _empty_result(symbol, name, error=last_error or "cache missing", expected_trade_date_value=expected_trade_date_value, error_category="cache_missing", attempt_logs=attempt_logs, attempt_count=attempt_count, last_error=last_error, fallback_reason="cache_missing", fetch_started_at=fetch_started_at, fetch_finished_at=fetch_finished_at, data_source_name="local_cache", source_attempts=source_attempts)
    df = _normalize_daily_columns(cached)
    trimmed_df, raw_latest_date, effective_latest_date, trimmed_count = _trim_to_expected_trade_date(df, expected_trade_date_value)
    if expected_trade_date_value is not None and effective_latest_date != expected_trade_date_value:
        msg = f"no expected trade date data in cache, raw_latest_date={raw_latest_date}, expected_trade_date={expected_trade_date_value}"
        attempt_logs.append({"attempt": 0, "error": msg, "error_category": "no_expected_trade_date_data", "final_source": "failed", "attempt_count": attempt_count, "last_error": last_error, "fallback_reason": "no_expected_trade_date_data"})
        result = _empty_result(symbol, name, error=msg, expected_trade_date_value=expected_trade_date_value, error_category="no_expected_trade_date_data", attempt_logs=attempt_logs, attempt_count=attempt_count, last_error=last_error, fallback_reason="no_expected_trade_date_data", fetch_started_at=fetch_started_at, fetch_finished_at=fetch_finished_at, data_source_name="local_cache", source_attempts=source_attempts)
        result.raw_latest_date = raw_latest_date
        result.effective_latest_date = effective_latest_date
        result.latest_date = effective_latest_date
        result.trimmed_future_rows_count = trimmed_count
        return result
    age = trade_day_diff(effective_latest_date, expected_trade_date_value, trade_dates)
    if age is None or age > max_cache_age_trade_days:
        msg = f"stale cache latest_date={effective_latest_date}, expected_trade_date={expected_trade_date_value}"
        attempt_logs.append({"attempt": 0, "error": msg, "error_category": "stale_cache", "final_source": "failed", "attempt_count": attempt_count, "last_error": last_error, "fallback_reason": "stale_cache"})
        return _empty_result(symbol, name, error=msg, expected_trade_date_value=expected_trade_date_value, error_category="stale_cache", attempt_logs=attempt_logs, attempt_count=attempt_count, last_error=last_error, fallback_reason="stale_cache", fetch_started_at=fetch_started_at, fetch_finished_at=fetch_finished_at, data_source_name="local_cache", source_attempts=source_attempts)
    attempt_logs.append({"attempt": 0, "error": last_error, "error_category": categorize_error(last_error), "final_source": "cache", "attempt_count": attempt_count, "last_error": last_error, "fallback_reason": fallback_reason})
    source_attempts.append(_source_attempt(source_name="local_cache", status="success", final_source="cache", error=last_error, error_category=categorize_error(last_error), raw_latest_date=raw_latest_date, effective_latest_date=effective_latest_date, rows=len(trimmed_df)))
    return _make_result(symbol, name, df, final_source="cache", expected_trade_date_value=expected_trade_date_value, error=last_error, error_category=categorize_error(last_error), attempt_logs=attempt_logs, attempt_count=attempt_count, last_error=last_error, fallback_reason=fallback_reason, fetch_started_at=fetch_started_at, fetch_finished_at=fetch_finished_at, data_source_name="local_cache", source_attempts=source_attempts, min_history_days=min_history_days)


def _fetch_stock_source(source_name: str, symbol: str, start_date: str, end_date: str, fetcher=None, network_timeout: float = 10.0) -> pd.DataFrame:
    if fetcher is not None and source_name == "injected":
        return fetcher(symbol=symbol, start_date=start_date, end_date=end_date)
    import akshare as ak
    _install_requests_default_timeout(network_timeout)

    if source_name == "akshare_eastmoney":
        return ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
    if source_name == "akshare_sina":
        return ak.stock_zh_a_daily(symbol=_ak_stock_symbol(symbol), start_date=start_date, end_date=end_date, adjust="qfq")
    raise ValueError(f"unsupported stock daily source: {source_name}")


def _fetch_index_source(source_name: str, index_code: str, ak_symbol: str, start_date: str, end_date: str, network_timeout: float = 10.0) -> pd.DataFrame:
    import akshare as ak
    _install_requests_default_timeout(network_timeout)

    if source_name == "akshare_sina":
        raw = ak.stock_zh_index_daily(symbol=ak_symbol)
        raw = raw.rename(columns={"date": "date", "open": "open", "close": "close", "high": "high", "low": "low", "volume": "volume"})
        raw["amount"] = raw.get("amount", 0)
        return raw
    if source_name == "akshare_eastmoney":
        return ak.index_zh_a_hist(symbol=_index_symbol_without_prefix(ak_symbol, index_code), period="daily", start_date=start_date, end_date=end_date)
    if source_name == "akshare_tencent":
        raw = ak.stock_zh_index_daily_tx(symbol=ak_symbol, start_date=start_date, end_date=end_date)
        raw["amount"] = raw.get("amount", 0)
        return raw
    raise ValueError(f"unsupported index daily source: {source_name}")


def fetch_stock_daily(
    symbol: str,
    name: str,
    days: int = 300,
    *,
    settings: dict | None = None,
    expected_trade_date_value: str | None = None,
    trade_dates: list[str] | None = None,
    sleep_fn=time.sleep,
    fetcher=None,
    cache_reader=read_cache,
    source_fetchers: dict[str, object] | None = None,
) -> FetchResult:
    ensure_directories()
    settings = settings or {}
    fetch_cfg = settings.get("data_fetch", {}) or {}
    cache_cfg = settings.get("cache", {}) or {}
    source_cfg = ((settings.get("data_sources", {}) or {}).get("stock_daily", {}) or {})
    max_retries = int(fetch_cfg.get("max_retries", 2))
    retry_backoff = list(fetch_cfg.get("retry_backoff_seconds", [1, 3]) or [1, 3])
    jitter_seconds = float(fetch_cfg.get("jitter_seconds", 0) or 0)
    max_cache_age = int(cache_cfg.get("max_cache_age_trade_days", 1))
    min_history_days = int(fetch_cfg.get("min_history_days", 120))
    network_timeout = float(fetch_cfg.get("network_timeout_seconds", 10) or 10)
    start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")
    end_date = datetime.now().strftime("%Y%m%d")
    attempt_logs: list[dict] = []
    source_attempts: list[dict] = []
    last_error = ""
    fetch_started_at = _now_text()
    attempt_count = 0
    if settings.get("akshare_runtime_unavailable") and fetcher is None and not source_fetchers:
        return _read_valid_cache(
            symbol,
            name,
            expected_trade_date_value=expected_trade_date_value,
            max_cache_age_trade_days=max_cache_age,
            trade_dates=trade_dates,
            last_error="AkShare startup preflight timed out",
            attempt_logs=attempt_logs,
            attempt_count=0,
            fetch_started_at=fetch_started_at,
            fetch_finished_at=_now_text(),
            cache_reader=cache_reader,
            min_history_days=min_history_days,
            source_attempts=source_attempts,
        )
    if fetcher is not None:
        source_names = ["injected"]
    else:
        primary = source_cfg.get("primary", "akshare_eastmoney")
        fallbacks = list(source_cfg.get("fallback", ["akshare_sina", "local_cache"]) or [])
        source_names = [primary] + [item for item in fallbacks if item != primary and item != "local_cache"]

    for source_name in source_names:
        source_fetcher = (source_fetchers or {}).get(source_name)
        for attempt in range(1, max_retries + 2):
            attempt_count += 1
            try:
                raw = source_fetcher(symbol=symbol, start_date=start_date, end_date=end_date) if source_fetcher else _fetch_stock_source(source_name, symbol, start_date, end_date, fetcher=fetcher, network_timeout=network_timeout)
                df = _normalize_daily_columns(raw).tail(days)
                _validate_daily(df)
                result = _make_result(
                    symbol,
                    name,
                    df,
                    final_source="fresh",
                    expected_trade_date_value=expected_trade_date_value,
                    attempt_logs=attempt_logs,
                    success_attempt=attempt_count,
                    attempt_count=attempt_count,
                    fetch_started_at=fetch_started_at,
                    fetch_finished_at=_now_text(),
                    data_source_name=source_name,
                    fallback_source_used="" if source_name == source_names[0] else source_name,
                    source_attempts=source_attempts,
                    min_history_days=min_history_days,
                )
                source_attempts.append(
                    _source_attempt(
                        source_name=source_name,
                        status="success" if result.final_source == "fresh" else "failed",
                        final_source=result.final_source,
                        error=result.error or result.data_staleness_reason or "",
                        error_category=result.error_category,
                        raw_latest_date=result.raw_latest_date,
                        effective_latest_date=result.effective_latest_date,
                        trimmed_future_rows_count=result.trimmed_future_rows_count,
                        rows=len(result.data),
                    )
                )
                attempt_logs.append({"attempt": attempt, "source_name": source_name, "error": result.error or result.data_staleness_reason or "", "error_category": result.error_category, "final_source": result.final_source, "attempt_count": attempt_count, "success_attempt": attempt_count if result.final_source == "fresh" else ""})
                if result.final_source == "fresh":
                    _write_cache(symbol, result.data)
                    return result
                last_error = result.error_category or result.data_staleness_reason or result.error or ""
                break
            except Exception as exc:  # pragma: no cover - network fallback depends on environment
                last_error = str(exc)
                error_category = categorize_error(exc)
                source_attempts.append(_source_attempt(source_name=source_name, status="failed", error=last_error, error_category=error_category))
                attempt_logs.append({"attempt": attempt, "source_name": source_name, "error": last_error, "error_category": error_category, "final_source": "", "attempt_count": attempt_count, "last_error": last_error})
                if attempt <= max_retries:
                    wait = float(retry_backoff[min(attempt - 1, len(retry_backoff) - 1)])
                    _sleep_with_jitter(wait, jitter_seconds, sleep_fn)

    return _read_valid_cache(
        symbol,
        name,
        expected_trade_date_value=expected_trade_date_value,
        max_cache_age_trade_days=max_cache_age,
        trade_dates=trade_dates,
        last_error=last_error,
        attempt_logs=attempt_logs,
        attempt_count=attempt_count,
        fetch_started_at=fetch_started_at,
        fetch_finished_at=_now_text(),
        cache_reader=cache_reader,
        min_history_days=min_history_days,
        source_attempts=source_attempts,
    )


def fetch_hs300_daily(days: int = 300) -> FetchResult:
    result = fetch_index_daily("000300", "\u6caa\u6df1300", "sh000300", days=days)
    if result.final_source == "fresh":
        result.data.to_csv(_index_cache_path(), index=False, encoding="utf-8-sig")
    return result


def fetch_index_daily(
    index_code: str,
    name: str,
    ak_symbol: str,
    days: int = 300,
    *,
    expected_trade_date_value: str | None = None,
    settings: dict | None = None,
    source_fetchers: dict[str, object] | None = None,
) -> FetchResult:
    ensure_directories()
    settings = settings or {}
    fetch_cfg = settings.get("data_fetch", {}) or {}
    source_cfg = ((settings.get("data_sources", {}) or {}).get("index_daily", {}) or {})
    min_history_days = int(fetch_cfg.get("min_history_days", 120))
    network_timeout = float(fetch_cfg.get("network_timeout_seconds", 10) or 10)
    path = _generic_index_cache_path(index_code)
    start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")
    end_date = datetime.now().strftime("%Y%m%d")
    primary = source_cfg.get("primary", "akshare_sina")
    fallbacks = list(source_cfg.get("fallback", ["akshare_eastmoney", "akshare_tencent", "local_cache"]) or [])
    source_names = [primary] + [item for item in fallbacks if item != primary and item != "local_cache"]
    if settings.get("akshare_runtime_unavailable") and not source_fetchers:
        source_names = []
    source_attempts: list[dict] = []
    attempt_logs: list[dict] = []
    last_error = "AkShare startup preflight timed out" if not source_names else ""
    for source_name in source_names:
        try:
            fetcher = (source_fetchers or {}).get(source_name)
            raw = fetcher(index_code=index_code, ak_symbol=ak_symbol, start_date=start_date, end_date=end_date) if fetcher else _fetch_index_source(source_name, index_code, ak_symbol, start_date, end_date, network_timeout=network_timeout)
            df = _normalize_daily_columns(raw).tail(days)
            _validate_daily(df, "index")
            result = _make_result(
                index_code,
                name,
                df,
                final_source="fresh",
                expected_trade_date_value=expected_trade_date_value,
                data_source_name=source_name,
                fallback_source_used="" if source_name == source_names[0] else source_name,
                source_attempts=source_attempts,
                min_history_days=min_history_days,
            )
            source_attempts.append(
                _source_attempt(
                    source_name=source_name,
                    status="success" if result.final_source == "fresh" else "failed",
                    final_source=result.final_source,
                    error=result.error or result.data_staleness_reason or "",
                    error_category=result.error_category,
                    raw_latest_date=result.raw_latest_date,
                    effective_latest_date=result.effective_latest_date,
                    trimmed_future_rows_count=result.trimmed_future_rows_count,
                    rows=len(result.data),
                )
            )
            attempt_logs.append({"attempt": 1, "source_name": source_name, "error": result.error or result.data_staleness_reason or "", "error_category": result.error_category, "final_source": result.final_source})
            result.attempt_logs = attempt_logs
            if result.final_source == "fresh":
                result.data.to_csv(path, index=False, encoding="utf-8-sig")
                return result
            last_error = result.error_category or result.data_staleness_reason or result.error or ""
        except Exception as exc:  # pragma: no cover - network fallback depends on environment
            last_error = str(exc)
            category = categorize_error(exc)
            source_attempts.append(_source_attempt(source_name=source_name, status="failed", error=last_error, error_category=category))
            attempt_logs.append({"attempt": 1, "source_name": source_name, "error": last_error, "error_category": category, "final_source": ""})

    if path.exists():
        cached = pd.read_csv(path)
        result = _make_result(
            index_code,
            name,
            _normalize_daily_columns(cached),
            final_source="cache",
            expected_trade_date_value=expected_trade_date_value,
            error=last_error,
            error_category=categorize_error(last_error),
            attempt_logs=attempt_logs,
            data_source_name="local_cache",
            source_attempts=source_attempts,
            min_history_days=min_history_days,
        )
        source_attempts.append(_source_attempt(source_name="local_cache", status="success" if result.final_source == "cache" else "failed", final_source=result.final_source, error=result.error or result.data_staleness_reason or "", error_category=result.error_category, raw_latest_date=result.raw_latest_date, effective_latest_date=result.effective_latest_date, trimmed_future_rows_count=result.trimmed_future_rows_count, rows=len(result.data)))
        return result
    source_attempts.append(_source_attempt(source_name="local_cache", status="failed", error="cache missing", error_category="cache_missing"))
    return _empty_result(index_code, name, error=last_error or "cache missing", expected_trade_date_value=expected_trade_date_value, error_category=categorize_error(last_error) if last_error else "cache_missing", attempt_logs=attempt_logs, data_source_name="local_cache", source_attempts=source_attempts)


def fetch_market_indices(expected_trade_date_value: str | None = None, settings: dict | None = None) -> list[dict]:
    specs = [
        {"index_code": "000300", "index_name": "\u6caa\u6df1300", "ak_symbol": "sh000300", "core": True},
        {"index_code": "000001", "index_name": "\u4e0a\u8bc1\u6307\u6570", "ak_symbol": "sh000001", "core": True},
        {"index_code": "399001", "index_name": "\u6df1\u8bc1\u6210\u6307", "ak_symbol": "sz399001", "core": True},
        {"index_code": "000510", "index_name": "\u4e2d\u8bc1A500", "ak_symbol": "sh000510", "core": False},
    ]
    results = []
    for spec in specs:
        result = fetch_index_daily(spec["index_code"], spec["index_name"], spec["ak_symbol"], expected_trade_date_value=expected_trade_date_value, settings=settings)
        results.append({**spec, "result": result})
    return results


def fetch_stocks(
    stock_pool: list[dict[str, str]],
    progress: bool = False,
    *,
    settings: dict | None = None,
    expected_trade_date_value: str | None = None,
    trade_dates: list[str] | None = None,
    sleep_fn=time.sleep,
    stock_fetcher=None,
) -> list[FetchResult]:
    stocks = []
    total = len(stock_pool)
    fetch_cfg = ((settings or {}).get("data_fetch", {}) or {})
    interval = float(fetch_cfg.get("request_interval_seconds", 0.5))
    jitter_seconds = float(fetch_cfg.get("jitter_seconds", 0) or 0)
    batch_size = int(fetch_cfg.get("batch_size", 0) or 0)
    batch_pause = float(fetch_cfg.get("batch_pause_seconds", 0) or 0)
    soft_limit = int(fetch_cfg.get("consecutive_failure_soft_limit", 0) or 0)
    soft_pause = float(fetch_cfg.get("consecutive_failure_soft_pause_seconds", 0) or 0)
    hard_limit = int(fetch_cfg.get("consecutive_failure_hard_limit", 0) or 0)
    stock_fetcher = stock_fetcher or fetch_stock_daily
    consecutive_network_failures = 0
    for idx, item in enumerate(stock_pool, start=1):
        if progress and (idx == 1 or idx % 10 == 0 or idx == total):
            print(f"获取日线进度：{idx}/{total} {item['symbol']} {item['name']}", flush=True)
        result = stock_fetcher(item["symbol"], item["name"], settings=settings, expected_trade_date_value=expected_trade_date_value, trade_dates=trade_dates)
        stocks.append(result)
        result_network_failed = result.final_source != "fresh" and (
            result.error_category == "network_error"
            or any(log.get("error_category") == "network_error" for log in (result.attempt_logs or []))
        )
        if result_network_failed:
            consecutive_network_failures += 1
        else:
            consecutive_network_failures = 0
        if soft_limit and consecutive_network_failures == soft_limit and soft_pause > 0:
            _sleep_with_jitter(soft_pause, jitter_seconds, sleep_fn)
        if hard_limit and consecutive_network_failures >= hard_limit:
            abort_reason = "too_many_consecutive_failures"
            for remaining in stock_pool[idx:]:
                stocks.append(
                    _empty_result(
                        remaining["symbol"],
                        remaining["name"],
                        error=abort_reason,
                        expected_trade_date_value=expected_trade_date_value,
                        error_category="data_fetch_aborted",
                        fallback_reason=abort_reason,
                        data_fetch_abort_reason=abort_reason,
                        fetch_started_at=_now_text(),
                        fetch_finished_at=_now_text(),
                    )
                )
            break
        if idx < total and interval > 0:
            _sleep_with_jitter(interval, jitter_seconds, sleep_fn)
        if batch_size > 0 and batch_pause > 0 and idx % batch_size == 0 and idx < total:
            _sleep_with_jitter(batch_pause, jitter_seconds, sleep_fn)
    return stocks


def summarize_stock_fetches(results: list[FetchResult]) -> dict:
    total = len(results)
    fresh = [r for r in results if r.final_source == "fresh" and r.is_expected_trade_date and not r.data.empty]
    cache = [r for r in results if r.final_source == "cache" and not r.data.empty]
    failed = [r for r in results if r.final_source == "failed" or r.data.empty]
    request_failed = [r for r in failed if r.error_category not in {"no_expected_trade_date_data", "stale_cache", "data_not_ready", "insufficient_history"}]
    no_expected = [r for r in results if r.error_category == "no_expected_trade_date_data"]
    data_not_ready = [r for r in results if r.error_category == "data_not_ready"]
    insufficient_history = [r for r in results if r.error_category == "insufficient_history"]
    source_unavailable = [r for r in results if r.error_category in {"network_error", "data_fetch_aborted", "cache_missing"}]
    ready_count = sum(1 for r in results if r.effective_latest_date and r.expected_trade_date and r.effective_latest_date == r.expected_trade_date)
    ready_ratio = ready_count / total if total else 0.0
    distribution: dict[str, int] = {}
    for result in results:
        effective_date = result.effective_latest_date or result.latest_date
        if not result.data.empty and effective_date:
            distribution[effective_date] = distribution.get(effective_date, 0) + 1
    market_data_date = ""
    if distribution:
        market_data_date = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))[0][0]
    unexpected = [r for r in results if not r.data.empty and not r.is_expected_trade_date]
    stale_cache = [r for r in results if r.error_category == "stale_cache"]
    future_trimmed = [r for r in results if int(r.trimmed_future_rows_count or 0) > 0]
    attempt_distribution = {
        "attempt_1_success": sum(1 for r in fresh if r.success_attempt == 1),
        "attempt_2_success": sum(1 for r in fresh if r.success_attempt == 2),
        "attempt_3_success": sum(1 for r in fresh if r.success_attempt == 3),
        "attempt_4_success": sum(1 for r in fresh if r.success_attempt == 4),
        "cache_fallback": len(cache),
    }
    network_error_count = sum(
        1
        for r in results
        if r.error_category == "network_error" or any(log.get("error_category") == "network_error" for log in (r.attempt_logs or []))
    )
    network_error_ratio = network_error_count / total if total else 0.0
    cache_fallback_ratio = len(cache) / total if total else 0.0
    data_fetch_abort_reason = next((r.data_fetch_abort_reason for r in results if r.data_fetch_abort_reason), "")
    if data_fetch_abort_reason:
        rerun_suggestion = "\u672c\u6b21\u6293\u53d6\u56e0\u8fde\u7eed\u7f51\u7edc\u5931\u8d25\u4e2d\u6b62\uff0c\u5efa\u8bae\u7a0d\u540e\u91cd\u8dd1\uff1b\u82e5\u591a\u6b21\u51fa\u73b0\uff0c\u8bf7\u5207\u6362\u5907\u7528\u6570\u636e\u6e90\u3002"
    elif cache_fallback_ratio > 0.20:
        rerun_suggestion = "\u7f13\u5b58\u56de\u9000\u6bd4\u4f8b\u8fc7\u9ad8\uff0c\u5efa\u8bae\u7a0d\u540e\u91cd\u8dd1\u3002"
    else:
        rerun_suggestion = "\u65e0\u9700\u56e0\u6570\u636e\u83b7\u53d6\u91cd\u8dd1\u3002"
    started_values = [r.fetch_started_at for r in results if r.fetch_started_at]
    finished_values = [r.fetch_finished_at for r in results if r.fetch_finished_at]
    fetch_started_at = min(started_values) if started_values else ""
    fetch_finished_at = max(finished_values) if finished_values else ""
    duration = 0.0
    if fetch_started_at and fetch_finished_at:
        try:
            duration = (datetime.strptime(fetch_finished_at, "%Y-%m-%d %H:%M:%S") - datetime.strptime(fetch_started_at, "%Y-%m-%d %H:%M:%S")).total_seconds()
        except ValueError:
            duration = 0.0
    return {
        "total_symbols": total,
        "fresh_success_count": len(fresh),
        "cache_fallback_count": len(cache),
        "failed_count": len(failed),
        "cache_fallback_ratio": cache_fallback_ratio,
        "request_failed_count": len(request_failed),
        "no_expected_trade_date_data_count": len(no_expected),
        "data_not_ready_count": len(data_not_ready),
        "insufficient_history_count": len(insufficient_history),
        "source_unavailable_count": len(source_unavailable),
        "unusable_for_signal_count": len(failed),
        "ready_count": ready_count,
        "ready_ratio": ready_ratio,
        "ready_ratio_text": f"{ready_count}/{total} = {ready_ratio:.1%}" if total else "0/0 = 0.0%",
        "fetch_started_at": fetch_started_at,
        "fetch_finished_at": fetch_finished_at,
        "fetch_duration_seconds": round(duration, 2),
        "attempt_success_distribution": attempt_distribution,
        "network_error_count": network_error_count,
        "network_error_ratio": round(network_error_ratio, 4),
        "data_fetch_abort_reason": data_fetch_abort_reason,
        "rerun_suggestion": rerun_suggestion,
        "fresh_success_ratio": len(fresh) / total if total else 0.0,
        "cache_fallback_symbols": [
            {"symbol": r.symbol, "name": r.name, "error": r.error or r.data_staleness_reason or "", "fallback_reason": r.fallback_reason, "attempt_count": r.attempt_count, "last_error": r.last_error, "data_source_name": r.data_source_name, "source_attempts": r.source_attempts or []}
            for r in cache
        ],
        "failed_symbols": [{"symbol": r.symbol, "name": r.name, "error": r.error or r.data_staleness_reason or "", "error_category": r.error_category, "data_source_name": r.data_source_name, "source_attempts": r.source_attempts or []} for r in failed],
        "unexpected_trade_date_count": len(unexpected),
        "stale_cache_count": len(stale_cache),
        "future_rows_trimmed_symbols_count": len(future_trimmed),
        "future_rows_trimmed_total_count": sum(int(r.trimmed_future_rows_count or 0) for r in future_trimmed),
        "latest_date_distribution": distribution,
        "market_data_date": market_data_date,
        "source_success_distribution": {
            source: sum(1 for r in fresh if r.data_source_name == source)
            for source in sorted({r.data_source_name for r in fresh if r.data_source_name})
        },
    }


def write_fetch_failure_log(results: list[FetchResult], report_date: str) -> Path | None:
    rows = []
    for result in results:
        for log in result.attempt_logs or []:
            rows.append(
                {
                    "date": report_date,
                    "symbol": result.symbol,
                    "name": result.name,
                    "data_type": "daily_price",
                    "attempt": log.get("attempt", ""),
                    "error": log.get("error", ""),
                    "error_category": log.get("error_category", ""),
                    "final_source": log.get("final_source") or result.final_source,
                    "attempt_count": log.get("attempt_count", result.attempt_count),
                    "last_error": log.get("last_error", result.last_error),
                    "fallback_reason": log.get("fallback_reason", result.fallback_reason),
                    "success_attempt": log.get("success_attempt", result.success_attempt or ""),
                    "data_fetch_abort_reason": result.data_fetch_abort_reason,
                    "raw_latest_date": result.raw_latest_date or "",
                    "effective_latest_date": result.effective_latest_date or "",
                    "data_source_name": result.data_source_name or "",
                    "fallback_source_used": result.fallback_source_used or "",
                    "source_attempts": result.source_attempts or [],
                }
            )
    if not rows:
        return None
    ensure_directories()
    path = LOG_DIR / f"{report_date}_fetch_failures.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def fetch_all(stock_pool: list[dict[str, str]], progress: bool = False) -> tuple[FetchResult, list[FetchResult]]:
    ensure_holdings_file()
    hs300 = fetch_hs300_daily()
    return hs300, fetch_stocks(stock_pool, progress)
