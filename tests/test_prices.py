"""Тесты этапа 5: нормализация дневных цен по выборке."""

from datetime import date

import pandas as pd
import pytest
import responses

from equity_factor_study.clients.moex import MOEXClient
from equity_factor_study.http import HttpClient
from equity_factor_study.pipelines.prices import (
    DAILY_COLUMNS,
    PricesError,
    collect_prices,
    normalize_history,
    summarize,
    write_daily_prices,
)

HISTORY_URL = (
    "https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities"
)

_COLUMNS = ["SECID", "BOARDID", "TRADEDATE", "CLOSE", "VALUE"]


def _history_payload(rows: list[list]) -> dict:
    return {
        "history": {"columns": _COLUMNS, "data": rows},
        "history.cursor": {"columns": ["INDEX", "TOTAL", "PAGESIZE"], "data": [[0, len(rows), 100]]},
    }


def _row(secid: str, date: str, close: float, value: float, board: str = "TQBR") -> list:
    return [secid, board, date, close, value]


def test_normalize_history_keeps_only_selected_board():
    table = pd.DataFrame(
        [
            _row("SBER", "2014-01-06", 98.91, 3154470383.9),
            _row("SBER", "2014-01-06", 999.0, 1.0, board="SMAL"),  # не тот режим
        ],
        columns=_COLUMNS,
    )
    normalized = normalize_history(table, board="TQBR")
    assert list(normalized.columns) == DAILY_COLUMNS
    assert len(normalized) == 1
    assert normalized.iloc[0]["close"] == pytest.approx(98.91)
    assert normalized.iloc[0]["trade_date"] == date(2014, 1, 6)


def test_normalize_history_drops_duplicates():
    table = pd.DataFrame(
        [
            _row("SBER", "2014-01-06", 98.91, 3154470383.9),
            _row("SBER", "2014-01-06", 99.0, 3154470383.9),
        ],
        columns=_COLUMNS,
    )
    normalized = normalize_history(table, board="TQBR")
    assert len(normalized) == 1


def test_normalize_history_handles_comma_decimal_and_missing():
    table = pd.DataFrame(
        [
            ["SBER", "TQBR", "2014-01-06", "98,91", "3154470383,9"],
            ["SBER", "TQBR", "2014-01-08", None, "123,45"],
        ],
        columns=_COLUMNS,
    )
    normalized = normalize_history(table, board="TQBR")
    assert normalized.iloc[0]["close"] == pytest.approx(98.91)
    assert normalized.iloc[0]["value"] == pytest.approx(3154470383.9)
    assert pd.isna(normalized.iloc[1]["close"])


def test_normalize_history_missing_required_column_raises():
    table = pd.DataFrame([{"SECID": "SBER", "CLOSE": 98.0}])
    with pytest.raises(PricesError, match="обязательные колонки"):
        normalize_history(table, board="TQBR")


def test_normalize_history_empty_returns_empty():
    normalized = normalize_history(pd.DataFrame(), board="TQBR")
    assert normalized.empty
    assert list(normalized.columns) == DAILY_COLUMNS


def test_summarize_reports_missing_trading_days():
    table = pd.DataFrame(
        [
            {"secid": "A", "trade_date": date(2014, 1, 6)},
            {"secid": "A", "trade_date": date(2014, 1, 8)},
            {"secid": "B", "trade_date": date(2014, 1, 6)},
            {"secid": "B", "trade_date": date(2014, 1, 7)},
            {"secid": "B", "trade_date": date(2014, 1, 8)},
        ]
    )
    summaries = summarize(table)
    by_secid = {item.secid: item for item in summaries}
    # Календарь = {06, 07, 08}; инструмент A не имеет 07.
    assert by_secid["A"].missing_trading_days == 1
    assert by_secid["A"].complete is False
    assert by_secid["B"].missing_trading_days == 0
    assert by_secid["B"].complete is True


@responses.activate
def test_collect_prices_end_to_end(tmp_path):
    secids = ["A", "B"]
    for secid in secids:
        responses.get(
            f"{HISTORY_URL}/{secid}.json",
            json=_history_payload([
                _row(secid, "2014-01-06", 10.0, 100.0),
                _row(secid, "2014-01-08", 11.0, 120.0),
            ]),
            status=200,
            match=[responses.matchers.query_param_matcher({"start": 0}, strict_match=False)],
        )
        responses.get(
            f"{HISTORY_URL}/{secid}.json",
            json=_history_payload([]),
            status=200,
            match=[responses.matchers.query_param_matcher({"start": 2}, strict_match=False)],
        )

    with HttpClient() as http:
        client = MOEXClient(http=http, raw_dir=tmp_path)
        result = collect_prices(
            client=client,
            secids=secids,
            board="TQBR",
            from_date=date(2014, 1, 1),
            to_date=date(2014, 1, 31),
        )

    assert result.rows == 4
    assert len(result.summaries) == 2
    assert set(result.table["secid"]) == {"A", "B"}


def test_write_daily_prices(tmp_path):
    table = pd.DataFrame(
        [
            {"secid": "A", "trade_date": date(2014, 1, 6), "close": 10.0, "value": 100.0},
        ],
        columns=DAILY_COLUMNS,
    )
    path = write_daily_prices(table, tmp_path / "daily_prices.csv")
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "2014-01-06" in content