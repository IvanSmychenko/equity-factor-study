import json
from datetime import date

import pytest
import responses

from equity_factor_study.clients.moex import MOEXClient
from equity_factor_study.http import HttpClient

HISTORY_URL = (
    "https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities/SBER.json"
)


@responses.activate
def test_history_page_is_saved_and_parsed(tmp_path):
    payload = {
        "history": {
            "columns": ["SECID", "BOARDID", "TRADEDATE", "CLOSE"],
            "data": [
                ["SBER", "TQBR", "2024-01-10", 270.0],
            ],
        },
        "history.cursor": {
            "columns": ["INDEX", "TOTAL", "PAGESIZE"],
            "data": [[0, 1, 100]],
        },
    }

    responses.get(HISTORY_URL, json=payload, status=200)

    with HttpClient() as http:
        client = MOEXClient(http=http, raw_dir=tmp_path)

        page = client.get_history_page(
            security="SBER",
            from_date=date(2024, 1, 10),
            till_date=date(2024, 1, 10),
        )

    assert page.table.to_dict("records") == [
        {
            "SECID": "SBER",
            "BOARDID": "TQBR",
            "TRADEDATE": "2024-01-10",
            "CLOSE": 270.0,
        }
    ]
    assert page.cursor.iloc[0]["TOTAL"] == 1

    assert json.loads(page.raw_path.read_bytes()) == payload

    metadata_path = page.raw_path.with_name(f"{page.raw_path.name}.metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["source"] == "MOEX ISS"
    assert metadata["request_parameters"]["from"] == "2024-01-10"
    assert metadata["request_parameters"]["start"] == 0
    assert len(responses.calls) == 1


@responses.activate
def test_empty_history_is_allowed(tmp_path):
    responses.get(
        HISTORY_URL,
        json={
            "history": {
                "columns": ["SECID", "TRADEDATE", "CLOSE"],
                "data": [],
            }
        },
        status=200,
    )

    with HttpClient() as http:
        client = MOEXClient(http=http, raw_dir=tmp_path)

        page = client.get_history_page(
            security="SBER",
            from_date=date(2024, 1, 1),
            till_date=date(2024, 1, 1),
        )

    assert page.table.empty
    assert list(page.table.columns) == ["SECID", "TRADEDATE", "CLOSE"]
    assert page.cursor.empty


@responses.activate
def test_invalid_payload_is_saved_before_parsing_error(tmp_path):
    responses.get(
        HISTORY_URL,
        json={"unexpected": "response"},
        status=200,
    )

    with HttpClient() as http:
        client = MOEXClient(http=http, raw_dir=tmp_path)

        with pytest.raises(ValueError, match="history"):
            client.get_history_page(
                security="SBER",
                from_date=date(2024, 1, 10),
                till_date=date(2024, 1, 10),
            )

    saved_files = [
        path for path in tmp_path.glob("*.json") if not path.name.endswith(".metadata.json")
    ]

    assert len(saved_files) == 1
    assert json.loads(saved_files[0].read_bytes()) == {"unexpected": "response"}


def test_invalid_date_range_is_rejected(tmp_path):
    with HttpClient() as http:
        client = MOEXClient(http=http, raw_dir=tmp_path)

        with pytest.raises(ValueError, match="from_date"):
            client.get_history_page(
                security="SBER",
                from_date=date(2024, 2, 1),
                till_date=date(2024, 1, 1),
            )
