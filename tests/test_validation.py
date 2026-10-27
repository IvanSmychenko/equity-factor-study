"""Тесты этапа 6: проверки качества (validation)."""

from datetime import date

import pandas as pd

from equity_factor_study.validation import (
    FINDING_COLUMNS,
    LARGE_DAY_CHANGE,
    NONPOSITIVE_PRICE,
    TRADING_GAP,
    VOLUME_DISCREPANCY,
    check_large_changes,
    check_missing_price_on_trading_day,
    check_nonpositive_prices,
    check_trading_gaps,
    check_volume_discrepancy,
)


def _prices(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["secid", "trade_date", "close", "value"])


def test_nonpositive_prices_detects_zero_and_negative():
    table = _prices(
        [
            {"secid": "A", "trade_date": date(2014, 1, 6), "close": 0.0, "value": 100.0},
            {"secid": "A", "trade_date": date(2014, 1, 7), "close": -5.0, "value": 100.0},
            {"secid": "A", "trade_date": date(2014, 1, 8), "close": 10.0, "value": 100.0},
        ]
    )
    findings = check_nonpositive_prices(table)
    assert len(findings) == 2
    assert set(findings["type"]) == {NONPOSITIVE_PRICE}
    assert set(findings["date"]) == {"2014-01-06", "2014-01-07"}


def test_nonpositive_prices_empty():
    assert check_nonpositive_prices(_prices([])).empty


def test_large_changes_detects_anomalous_move():
    table = _prices(
        [
            {"secid": "A", "trade_date": date(2014, 1, 6), "close": 10.0, "value": 1.0},
            {"secid": "A", "trade_date": date(2014, 1, 7), "close": 100.0, "value": 1.0},  # 900%
            {"secid": "A", "trade_date": date(2014, 1, 8), "close": 100.0, "value": 1.0},
        ]
    )
    findings = check_large_changes(table)
    assert len(findings) == 1
    assert findings.iloc[0]["type"] == LARGE_DAY_CHANGE
    assert findings.iloc[0]["date"] == "2014-01-07"


def test_large_changes_respects_event_dates():
    table = _prices(
        [
            {"secid": "A", "trade_date": date(2014, 1, 6), "close": 10.0, "value": 1.0},
            {"secid": "A", "trade_date": date(2014, 1, 7), "close": 100.0, "value": 1.0},
        ]
    )
    # Событие на 2014-01-07 -> большой скачок не считается дефектом.
    event_dates = {("A", "2014-01-07")}
    findings = check_large_changes(table, event_dates=event_dates)
    assert findings.empty


def test_trading_gaps_detects_long_gap():
    table = _prices(
        [
            {"secid": "A", "trade_date": date(2014, 1, 6), "close": 10.0, "value": 1.0},
            {"secid": "A", "trade_date": date(2014, 2, 6), "close": 10.0, "value": 1.0},  # 31 день
        ]
    )
    findings = check_trading_gaps(table, max_gap_days=10)
    assert len(findings) == 1
    assert findings.iloc[0]["type"] == TRADING_GAP
    assert findings.iloc[0]["value"] == 31


def test_volume_discrepancy_detects_mismatch():
    table = _prices(
        [
            {"secid": "A", "trade_date": date(2014, 1, 6), "close": 10.0, "value": 100.0},
            {"secid": "A", "trade_date": date(2014, 1, 7), "close": 10.0, "value": 100.0},
        ]
    )
    expected = {"A": 100.0}  # сумма = 200.0, ожидание 100.0 -> большое отклонение
    findings = check_volume_discrepancy(table, expected_volume=expected, tolerance=0.01)
    assert len(findings) == 1
    assert findings.iloc[0]["type"] == VOLUME_DISCREPANCY


def test_volume_discrepancy_skipped_without_expected():
    table = _prices([{"secid": "A", "trade_date": date(2014, 1, 6), "close": 10.0, "value": 100.0}])
    assert check_volume_discrepancy(table, expected_volume=None).empty


def test_missing_price_on_trading_day_detects_gap():
    # Календарь = {13,14,15}. У инструмента A пустая цена на 14 (пропуск).
    # У B данные заканчиваются 14.01, поэтому 15.01 — вне его интервала торгов
    # (инструмент уже не обращается), а не пропуск.
    table = _prices(
        [
            {"secid": "A", "trade_date": date(2014, 1, 13), "close": 10.0, "value": 1.0},
            {"secid": "A", "trade_date": date(2014, 1, 14), "close": None, "value": 1.0},
            {"secid": "A", "trade_date": date(2014, 1, 15), "close": 10.0, "value": 1.0},
            {"secid": "B", "trade_date": date(2014, 1, 13), "close": 10.0, "value": 1.0},
            {"secid": "B", "trade_date": date(2014, 1, 14), "close": 10.0, "value": 1.0},
        ]
    )
    calendar = {date(2014, 1, 13), date(2014, 1, 14), date(2014, 1, 15)}
    findings = check_missing_price_on_trading_day(table, reference_calendar=calendar)
    # Замечание одно: A @ 14.01 (пустая цена). B @ 15.01 не считается,
    # т.к. это вне интервала торгов B.
    assert len(findings) == 1
    assert set(findings["secid"]) == {"A"}


def test_missing_price_detects_gap_within_two_instruments():
    # Календарь = {13,14,15}; оба инструмента торгуются весь интервал.
    table = _prices(
        [
            {"secid": "A", "trade_date": date(2014, 1, 13), "close": 10.0, "value": 1.0},
            {"secid": "A", "trade_date": date(2014, 1, 14), "close": None, "value": 1.0},
            {"secid": "A", "trade_date": date(2014, 1, 15), "close": 10.0, "value": 1.0},
            {"secid": "B", "trade_date": date(2014, 1, 13), "close": 10.0, "value": 1.0},
            {"secid": "B", "trade_date": date(2014, 1, 15), "close": 10.0, "value": 1.0},
        ]
    )
    calendar = {date(2014, 1, 13), date(2014, 1, 14), date(2014, 1, 15)}
    findings = check_missing_price_on_trading_day(table, reference_calendar=calendar)
    # A @ 14.01 (пустая), B @ 14.01 (нет строки).
    assert len(findings) == 2
    assert set(findings["secid"]) == {"A", "B"}


def test_missing_price_ignores_pre_debut_days():
    """Дни до первой котировки инструмента не считаются пропусками цены."""
    # Инструмент A торгуется только с 14.01; календарь начинается с 13.01.
    table = _prices(
        [
            {"secid": "A", "trade_date": date(2014, 1, 14), "close": 10.0, "value": 1.0},
            {"secid": "A", "trade_date": date(2014, 1, 15), "close": 10.0, "value": 1.0},
        ]
    )
    calendar = {date(2014, 1, 13), date(2014, 1, 14), date(2014, 1, 15)}
    findings = check_missing_price_on_trading_day(table, reference_calendar=calendar)
    # 13.01 — не пропуск (инструмент ещё не торговался); 14,15 — есть цена.
    assert len(findings) == 0


def test_missing_price_ignores_ru_holidays():
    """Новогодние каникулы (нерабочие дни) не считаются пропусками цены."""
    table = _prices(
        [
            {"secid": "A", "trade_date": date(2014, 1, 13), "close": 10.0, "value": 1.0},
            {"secid": "A", "trade_date": date(2014, 1, 14), "close": 10.0, "value": 1.0},
        ]
    )
    # Календарь включает праздничные каникулы 6–8.01 и рабочие 13–14.01.
    calendar = {date(2014, 1, 6), date(2014, 1, 7), date(2014, 1, 8), date(2014, 1, 13), date(2014, 1, 14)}
    findings = check_missing_price_on_trading_day(table, reference_calendar=calendar)
    # 6–8.01 — нерабочие (каникулы), не пропуски; 13,14 — есть цена.
    assert len(findings) == 0


def test_finding_columns():
    table = _prices([{"secid": "A", "trade_date": date(2014, 1, 6), "close": 0.0, "value": 1.0}])
    findings = check_nonpositive_prices(table)
    assert list(findings.columns) == FINDING_COLUMNS