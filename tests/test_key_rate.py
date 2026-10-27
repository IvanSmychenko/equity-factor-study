"""Тесты этапа 5: нормализация ряда ключевой ставки."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import responses

from equity_factor_study.clients.cbr import CBR_ENDPOINT, CBRClient
from equity_factor_study.http_client import HttpClient
from equity_factor_study.pipelines.key_rate import (
    DAILY_COLUMNS,
    collect_key_rate,
    normalize_key_rate,
    summarize_key_rate,
    write_daily_key_rate,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _cbr_payload() -> bytes:
    return (FIXTURES / "cbr_keyrate_2014-01.xml").read_bytes()


def test_normalize_key_rate_types():
    table = pd.DataFrame(
        [
            {"date": date(2014, 1, 10), "rate": "5.50"},
            {"date": date(2014, 1, 9), "rate": 5.50},
            {"date": date(2014, 1, 10), "rate": 5.50},  # дубль даты
        ]
    )
    normalized = normalize_key_rate(table)
    assert list(normalized.columns) == DAILY_COLUMNS
    assert len(normalized) == 2
    assert normalized["date"].iloc[0] == date(2014, 1, 9)
    assert normalized["rate"].iloc[0] == pytest.approx(5.5)
    assert normalized["date"].is_monotonic_increasing


def test_normalize_key_rate_empty():
    normalized = normalize_key_rate(pd.DataFrame())
    assert normalized.empty
    assert list(normalized.columns) == DAILY_COLUMNS


def test_summarize_key_rate_coverage():
    # Будни 1..10.01.2014, наблюдения только на 6,7,8 -> пропуски на 1..5,9,10.
    table = pd.DataFrame(
        {
            "date": [date(2014, 1, 6), date(2014, 1, 7), date(2014, 1, 8)],
            "rate": [5.5, 5.5, 5.5],
        }
    )
    summary = summarize_key_rate(table, date(2014, 1, 1), date(2014, 1, 10))
    assert summary.rows == 3
    assert summary.date_from == date(2014, 1, 6)
    assert summary.date_to == date(2014, 1, 8)
    assert summary.gaps > 0
    assert summary.complete is False


def test_summarize_key_rate_complete():
    table = pd.DataFrame(
        {
            "date": [date(2014, 1, 6), date(2014, 1, 7), date(2014, 1, 8)],
            "rate": [5.5, 5.5, 5.5],
        }
    )
    summary = summarize_key_rate(table, date(2014, 1, 6), date(2014, 1, 8))
    assert summary.gaps == 0
    assert summary.complete is True


def test_summarize_key_rate_holidays_not_gaps():
    """Праздники и переносы РФ не считаются разрывами покрытия.

    Период с 2014-01-01 по 2014-01-10: новогодние каникулы (1–8 янв) —
    нерабочие; рабочие будни — только 9 и 10 января. Наблюдения есть
    на 9 и 10 -> разрывов нет.
    """
    table = pd.DataFrame(
        {
            "date": [date(2014, 1, 9), date(2014, 1, 10)],
            "rate": [5.5, 5.5],
        }
    )
    summary = summarize_key_rate(table, date(2014, 1, 1), date(2014, 1, 10))
    assert summary.gaps == 0
    assert summary.complete is True


@responses.activate
def test_collect_key_rate_end_to_end(tmp_path):
    responses.post(CBR_ENDPOINT, body=_cbr_payload(), status=200, content_type="text/xml")

    with HttpClient() as http:
        client = CBRClient(http=http, raw_dir=tmp_path)
        result = collect_key_rate(client, from_date=date(2014, 1, 1), to_date=date(2014, 1, 31))

    assert result.rows == 17
    assert result.summary.date_to == date(2014, 1, 31)
    assert len(result.raw_paths) == 1


def test_write_daily_key_rate(tmp_path):
    table = pd.DataFrame(
        {"date": [date(2014, 1, 9)], "rate": [5.5]},
        columns=DAILY_COLUMNS,
    )
    path = write_daily_key_rate(table, tmp_path / "key_rate.csv")
    assert path.exists()
    assert "2014-01-09" in path.read_text(encoding="utf-8")