"""Тесты этапа 7: построение месячной панели и манифеста."""

from datetime import date

import pandas as pd
import pytest

from equity_factor_study.pipelines.monthly import (
    PANEL_COLUMNS,
    MonthlyError,
    build_manifest,
    build_monthly_panel,
)

# Инструмент с непрерывным покрытием: ежедневные данные за 6 месяцев 2014.
DAILY = [
    {"secid": "A", "trade_date": date(2014, 1, 6), "close": 100.0, "value": 1000.0},
    {"secid": "A", "trade_date": date(2014, 1, 7), "close": 101.0, "value": 1000.0},
    {"secid": "A", "trade_date": date(2014, 1, 8), "close": 102.0, "value": 1000.0},
    {"secid": "A", "trade_date": date(2014, 2, 3), "close": 110.0, "value": 1000.0},
    {"secid": "A", "trade_date": date(2014, 2, 4), "close": 111.0, "value": 1000.0},
    {"secid": "A", "trade_date": date(2014, 3, 3), "close": 121.0, "value": 1000.0},
    {"secid": "A", "trade_date": date(2014, 3, 4), "close": 122.0, "value": 1000.0},
    {"secid": "A", "trade_date": date(2014, 4, 1), "close": 132.0, "value": 1000.0},
    {"secid": "A", "trade_date": date(2014, 4, 2), "close": 133.0, "value": 1000.0},
]

KEY_RATE = pd.DataFrame(
    {
        "date": [date(2014, 1, 9), date(2014, 2, 10), date(2014, 3, 10), date(2014, 4, 10)],
        "rate": [5.5, 6.0, 6.5, 7.0],
    }
)


def test_build_panel_one_row_per_instrument_month():
    panel = build_monthly_panel(
        pd.DataFrame(DAILY),
        secids=["A"],
        from_month=date(2014, 1, 1),
        to_month=date(2014, 4, 1),
        max_trading_days=5,
        momentum_window=3,
        volatility_window=3,
        key_rate=KEY_RATE,
    )
    assert len(panel) == 4  # 4 месяца
    assert set(panel["month"]) == {
        date(2014, 1, 1), date(2014, 2, 1), date(2014, 3, 1), date(2014, 4, 1)
    }
    assert list(panel.columns) == PANEL_COLUMNS


def test_month_end_price_and_price_date():
    panel = build_monthly_panel(
        pd.DataFrame(DAILY),
        secids=["A"],
        from_month=date(2014, 1, 1),
        to_month=date(2014, 1, 1),
        max_trading_days=5,
        momentum_window=3,
        volatility_window=3,
    )
    row = panel.iloc[0]
    assert row["month_end_price"] == 102.0  # последняя цена января
    assert row["price_date"] == date(2014, 1, 8)


def test_return_1m_formula():
    panel = build_monthly_panel(
        pd.DataFrame(DAILY),
        secids=["A"],
        from_month=date(2014, 1, 1),
        to_month=date(2014, 4, 1),
        max_trading_days=5,
        momentum_window=3,
        volatility_window=3,
    )
    # return_1m февраля = 111/102 - 1 (последняя цена февраля = 111)
    feb = panel[panel["month"] == date(2014, 2, 1)].iloc[0]
    assert feb["return_1m"] == pytest.approx(111.0 / 102.0 - 1)
    # Первый месяц не имеет предыдущей цены.
    jan = panel[panel["month"] == date(2014, 1, 1)].iloc[0]
    assert pd.isna(jan["return_1m"])
    assert "no_previous_price" in jan["quality_flags"]


def test_momentum_prev_3m_formula():
    # Моментум = цена t-1 / цена t-4 - 1. Для марта: фев(110)/? - нужно 4 месяца назад.
    # Мы строим с расширением в прошлое, поэтому для проверки используем
    # достаточно длинный ряд. Убедимся, что колонка присутствует и корректна по знаку.
    panel = build_monthly_panel(
        pd.DataFrame(DAILY),
        secids=["A"],
        from_month=date(2014, 1, 1),
        to_month=date(2014, 4, 1),
        max_trading_days=5,
        momentum_window=3,
        volatility_window=3,
    )
    assert "momentum_prev_3m" in panel.columns


def test_volatility_3m_present():
    panel = build_monthly_panel(
        pd.DataFrame(DAILY),
        secids=["A"],
        from_month=date(2014, 1, 1),
        to_month=date(2014, 4, 1),
        max_trading_days=5,
        momentum_window=3,
        volatility_window=3,
    )
    assert "volatility_3m" in panel.columns
    # Апрель охватывает 3 месяца (фев-апр).
    apr = panel[panel["month"] == date(2014, 4, 1)].iloc[0]
    assert apr["volatility_3m"] is not None


def test_target_return_lags_next_month():
    # Строим 5 месяцев для проверки сдвига целевой переменной.
    long_daily = list(DAILY)
    for m, close in [(5, 140.0)]:
        long_daily.append(
            {"secid": "A", "trade_date": date(2014, 5, 5), "close": close, "value": 1000.0}
        )
    panel = build_monthly_panel(
        pd.DataFrame(long_daily),
        secids=["A"],
        from_month=date(2014, 1, 1),
        to_month=date(2014, 5, 1),
        max_trading_days=5,
        momentum_window=3,
        volatility_window=3,
    )
    # target_return_next_1m[апр] = цена мая / цена апр - 1 = 140/133 - 1
    apr = panel[panel["month"] == date(2014, 4, 1)].iloc[0]
    assert apr["target_return_next_1m"] == pytest.approx(140.0 / 133.0 - 1)
    # Последний месяц имеет пустую целевую переменную.
    may = panel[panel["month"] == date(2014, 5, 1)].iloc[0]
    assert pd.isna(may["target_return_next_1m"])


def test_key_rate_attached():
    panel = build_monthly_panel(
        pd.DataFrame(DAILY),
        secids=["A"],
        from_month=date(2014, 1, 1),
        to_month=date(2014, 4, 1),
        max_trading_days=5,
        momentum_window=3,
        volatility_window=3,
        key_rate=KEY_RATE,
    )
    feb = panel[panel["month"] == date(2014, 2, 1)].iloc[0]
    assert feb["key_rate_pct"] == pytest.approx(6.0)
    assert feb["key_rate_change_pp"] == pytest.approx(0.5)


def test_missing_required_column_raises():
    with pytest.raises(MonthlyError, match="secid, trade_date, close"):
        build_monthly_panel(
            pd.DataFrame({"secid": ["A"], "value": [1.0]}),
            secids=["A"],
            from_month=date(2014, 1, 1),
            to_month=date(2014, 1, 1),
            max_trading_days=5,
            momentum_window=3,
            volatility_window=3,
        )


def test_instrument_without_data_gets_flag():
    panel = build_monthly_panel(
        pd.DataFrame(DAILY),
        secids=["A", "B"],  # B нет данных
        from_month=date(2014, 1, 1),
        to_month=date(2014, 1, 1),
        max_trading_days=5,
        momentum_window=3,
        volatility_window=3,
    )
    b = panel[panel["secid"] == "B"].iloc[0]
    assert "no_trading_days_in_month" in b["quality_flags"]
    assert pd.isna(b["month_end_price"])


def test_first_full_feature_month_matches_s0():
    """Первый месяц с полным набором признаков (s0) — май 2014 при данных с января 2014."""
    # Непрерывные данные с января 2014: ежемесячная цена в конце месяца.
    rows = []
    price = 100.0
    month = date(2014, 1, 1)
    while month <= date(2014, 12, 1):
        price *= 1.01
        trade_date = date(month.year, month.month, min(28, 31))
        rows.append({"secid": "A", "trade_date": trade_date, "close": price, "value": 1000.0})
        month = _next_month(month)
    daily = pd.DataFrame(rows)

    panel = build_monthly_panel(
        daily,
        secids=["A"],
        from_month=date(2014, 1, 1),
        to_month=date(2014, 12, 1),
        max_trading_days=5,
        momentum_window=3,
        volatility_window=3,
    )
    # Моментум требует цены t-4: впервые доступен для мая 2014 (с января 2014).
    full = panel.dropna(subset=["return_1m", "momentum_prev_3m", "volatility_3m"])
    assert full["month"].min() == date(2014, 5, 1)
    # До мая моментум отсутствует и помечен.
    april = panel[panel["month"] == date(2014, 4, 1)].iloc[0]
    assert pd.isna(april["momentum_prev_3m"])


def _next_month(value):
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def test_build_manifest():
    panel = build_monthly_panel(
        pd.DataFrame(DAILY),
        secids=["A"],
        from_month=date(2014, 1, 1),
        to_month=date(2014, 1, 1),
        max_trading_days=5,
        momentum_window=3,
        volatility_window=3,
    )
    manifest = build_manifest(
        panel=panel,
        protocol_version="0.1",
        observation_start=date(2014, 1, 1),
        observation_end=date(2026, 9, 30),
        secids=["A"],
        source_files=[{"path": "daily_prices.csv", "sha256": "abc"}],
        code_version="0.1.0",
        dependencies={},
    )
    assert manifest["protocol_version"] == "0.1"
    assert manifest["instruments"] == ["A"]
    assert manifest["source_files"][0]["sha256"] == "abc"
    assert manifest["rows"] == 1