from src.data_fetcher import is_main_board_symbol, is_risk_stock_name


def test_main_board_symbol_detection():
    assert is_main_board_symbol("600000")
    assert is_main_board_symbol("000001")
    assert is_main_board_symbol("002475")
    assert not is_main_board_symbol("300750")
    assert not is_main_board_symbol("688981")
    assert not is_main_board_symbol("430047")


def test_risk_stock_name_detection():
    assert is_risk_stock_name("*ST测试")
    assert is_risk_stock_name("退市测试")
    assert not is_risk_stock_name("招商银行")
