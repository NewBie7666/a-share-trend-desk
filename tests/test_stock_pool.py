import pandas as pd

from src.data_fetcher import _normalize_spot_columns, is_main_board_symbol, is_risk_stock_name


def test_main_board_symbol_detection():
    assert is_main_board_symbol("600000")
    assert is_main_board_symbol("000001")
    assert is_main_board_symbol("002475")
    assert not is_main_board_symbol("300750")
    assert not is_main_board_symbol("688981")
    assert not is_main_board_symbol("430047")


def test_main_board_symbol_detection_accepts_exchange_prefixes():
    assert is_main_board_symbol("sh600000")
    assert is_main_board_symbol("sz000001")
    assert not is_main_board_symbol("bj920001")


def test_legacy_spot_schema_uses_positional_fallback_and_strips_prefix():
    columns = [f"broken_{index}" for index in range(14)]
    row = ["sh600000", "浦发银行", 10.5, 0.1, 1.0, 0, 0, 0, 0, 0, 0, 12345, 67890000, "14:30:00"]

    normalized = _normalize_spot_columns(pd.DataFrame([row], columns=columns))

    assert normalized.loc[0, "symbol"] == "600000"
    assert normalized.loc[0, "name"] == "浦发银行"
    assert normalized.loc[0, "price"] == 10.5
    assert normalized.loc[0, "amount"] == 67890000


def test_risk_stock_name_detection():
    assert is_risk_stock_name("*ST测试")
    assert is_risk_stock_name("退市测试")
    assert not is_risk_stock_name("招商银行")
