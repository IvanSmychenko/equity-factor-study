import json
from datetime import date

import pytest
import responses

from equity_factor_study.clients.moex import MOEXClient, MOEXHistoryError
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


# --- полная пагинация (этап 3) -------------------------------------------------


def _page_payload(rows: list[list], *, index: int, total: int, page_size: int) -> dict:
    """Сформировать JSON-ответ ISS для одной страницы истории."""
    columns = ["SECID", "BOARDID", "TRADEDATE", "CLOSE"]
    return {
        "history": {"columns": columns, "data": rows},
        "history.cursor": {
            "columns": ["INDEX", "TOTAL", "PAGESIZE"],
            "data": [[index, total, page_size]],
        },
    }


def _register_pages(pages: list[dict], *, starts: list[int]) -> None:
    """Зарегистрировать ответы для последовательных start в responses."""
    for start, payload in zip(starts, pages):
        responses.get(
            HISTORY_URL,
            json=payload,
            status=200,
            match=[responses.matchers.query_param_matcher({"start": start}, strict_match=False)],
        )


@responses.activate
def test_full_history_three_pages(tmp_path):
    page_size = 2
    total = 5
    pages = [
        _page_payload(
            [["SBER", "TQBR", f"2024-01-0{i}", float(i)] for i in range(1, 3)],
            index=0,
            total=total,
            page_size=page_size,
        ),
        _page_payload(
            [["SBER", "TQBR", f"2024-01-0{i}", float(i)] for i in range(3, 5)],
            index=2,
            total=total,
            page_size=page_size,
        ),
        _page_payload(
            [["SBER", "TQBR", "2024-01-05", 5.0]],
            index=4,
            total=total,
            page_size=page_size,
        ),
    ]
    _register_pages(pages, starts=[0, 2, 4])

    with HttpClient() as http:
        client = MOEXClient(http=http, raw_dir=tmp_path)
        download = client.get_full_history(
            security="SBER",
            from_date=date(2024, 1, 1),
            till_date=date(2024, 1, 5),
        )

    assert download.rows == 5
    assert download.pages == 3
    assert download.page_sizes == [2, 2, 1]
    assert download.cursor_total == 5
    assert download.date_interval == (date(2024, 1, 1), date(2024, 1, 5))
    assert len(responses.calls) == 3

    # Каждая страница сохранена отдельным файлом с метаданными.
    saved = [p for p in tmp_path.glob("*.json") if not p.name.endswith(".metadata.json")]
    assert len(saved) == 3


@responses.activate
def test_full_history_sum_matches_pages(tmp_path):
    """Сумма строк по страницам совпадает с итогом (критерий приёмки этапа 3)."""
    pages = [
        _page_payload([[s + "_TQBR", "TQBR", f"2024-01-0{i}", float(i)] for i, s in enumerate(["A", "B"])],
                      index=0, total=4, page_size=2),
        _page_payload([[s + "_TQBR", "TQBR", f"2024-01-0{i}", float(i)] for i, s in enumerate(["C", "D"])],
                      index=2, total=4, page_size=2),
    ]
    _register_pages(pages, starts=[0, 2])

    with HttpClient() as http:
        client = MOEXClient(http=http, raw_dir=tmp_path)
        download = client.get_full_history(
            security="SBER",
            from_date=date(2024, 1, 1),
            till_date=date(2024, 1, 4),
        )

    assert download.rows == sum(download.page_sizes) == 4
    assert download.pages == 2


@responses.activate
def test_full_history_empty_is_allowed(tmp_path):
    responses.get(
        HISTORY_URL,
        json={
            "history": {"columns": ["SECID", "TRADEDATE", "CLOSE"], "data": []}
        },
        status=200,
        match=[responses.matchers.query_param_matcher({"start": 0}, strict_match=False)],
    )

    with HttpClient() as http:
        client = MOEXClient(http=http, raw_dir=tmp_path)
        download = client.get_full_history(
            security="SBER",
            from_date=date(2024, 1, 1),
            till_date=date(2024, 1, 1),
        )

    assert download.rows == 0
    assert download.pages == 0
    assert download.date_interval == (None, None)
    assert not download


@responses.activate
def test_full_history_non_advancing_cursor_raises(tmp_path):
    """Защита от зацикливания: курсор возвращается к прежнему INDEX."""
    page = _page_payload(
        [["SBER", "TQBR", "2024-01-01", 1.0]],
        index=0,
        total=5,
        page_size=1,
    )
    # Разные start, но курсор всегда сообщает INDEX=0 => позиция не двигается.
    responses.get(
        HISTORY_URL,
        json=page,
        status=200,
        match=[responses.matchers.query_param_matcher({"start": 0}, strict_match=False)],
    )
    responses.get(
        HISTORY_URL,
        json=page,
        status=200,
        match=[responses.matchers.query_param_matcher({"start": 1}, strict_match=False)],
    )

    with HttpClient() as http:
        client = MOEXClient(http=http, raw_dir=tmp_path)

        with pytest.raises(MOEXHistoryError, match="не продвигает"):
            client.get_full_history(
                security="SBER",
                from_date=date(2024, 1, 1),
                till_date=date(2024, 1, 5),
            )


@responses.activate
def test_full_history_duplicate_detected(tmp_path):
    """Дубли по паре «инструмент — торговая дата» обнаруживаются (этап 3)."""
    pages = [
        _page_payload(
            [["SBER", "TQBR", "2024-01-01", 1.0], ["SBER", "TQBR", "2024-01-02", 2.0]],
            index=0,
            total=3,
            page_size=2,
        ),
        _page_payload(
            [["SBER", "TQBR", "2024-01-01", 99.0]],  # дубль даты
            index=2,
            total=3,
            page_size=2,
        ),
    ]
    _register_pages(pages, starts=[0, 2])

    with HttpClient() as http:
        client = MOEXClient(http=http, raw_dir=tmp_path)

        with pytest.raises(MOEXHistoryError, match="дубли"):
            client.get_full_history(
                security="SBER",
                from_date=date(2024, 1, 1),
                till_date=date(2024, 1, 3),
            )


@responses.activate
def test_full_history_incomplete_relative_to_cursor(tmp_path):
    """Неполнота набора относительно курсора обнаруживается (этап 3)."""
    # total=3, но фактически приходит 2 строки => набор неполный.
    pages = [
        _page_payload(
            [["SBER", "TQBR", "2024-01-01", 1.0]],
            index=0,
            total=3,
            page_size=1,
        ),
        _page_payload(
            [["SBER", "TQBR", "2024-01-02", 2.0]],
            index=1,
            total=3,
            page_size=2,
        ),
    ]
    _register_pages(pages, starts=[0, 1])

    with HttpClient() as http:
        client = MOEXClient(http=http, raw_dir=tmp_path)

        with pytest.raises(MOEXHistoryError, match="Неполная история"):
            client.get_full_history(
                security="SBER",
                from_date=date(2024, 1, 1),
                till_date=date(2024, 1, 3),
            )
