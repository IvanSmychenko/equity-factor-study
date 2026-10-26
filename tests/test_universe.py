"""Тесты этапа 4: справочники и историческая совокупность."""


import tomllib

import pandas as pd
import pytest
import responses

from equity_factor_study.clients.moex import MOEXClient
from equity_factor_study.http_client import HttpClient
from equity_factor_study.pipelines.catalog import CatalogResult, load_catalog
from equity_factor_study.pipelines.universe import (
    UniverseError,
    build_and_write_universe,
    build_universe,
    load_lock_file,
    write_exclusions,
    write_lock_file,
)

HISTORY_URL = (
    "https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities"
)
CATALOG_URL = (
    "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
)


def _catalog_payload(rows: list[list]) -> dict:
    columns = [
        "SECID", "BOARDID", "SHORTNAME", "SECNAME", "SECTYPE",
        "LISTLEVEL", "ISIN", "LATNAME", "STATUS",
    ]
    return {"securities": {"columns": columns, "data": rows}}


def _catalog_row(secid: str, sectype: str = "1", isin: str = "") -> list:
    return [secid, "TQBR", secid, f"{secid} ао", sectype, 1, isin, secid, "A"]


def _history_payload(secid: str, value: float, on_date: str = "2014-06-16") -> dict:
    columns = ["SECID", "BOARDID", "TRADEDATE", "VALUE", "CLOSE"]
    return {
        "history": {"columns": columns, "data": [[secid, "TQBR", on_date, value, 1.0]]},
        "history.cursor": {"columns": ["INDEX", "TOTAL", "PAGESIZE"], "data": [[0, 1, 100]]},
    }


# --- справочник ---------------------------------------------------------------


@responses.activate
def test_load_catalog_filters_common_shares(tmp_path):
    responses.get(
        CATALOG_URL,
        json=_catalog_payload([
            _catalog_row("SBER", isin="RU0009029540"),
            _catalog_row("SBERP", sectype="2", isin="RU0009029557"),
            _catalog_row("LKOH", isin="RU0009024277"),
        ]),
        status=200,
        match=[responses.matchers.query_param_matcher({"start": 0}, strict_match=False)],
    )
    responses.get(
        CATALOG_URL,
        json=_catalog_payload([]),
        status=200,
        match=[responses.matchers.query_param_matcher({"start": 3}, strict_match=False)],
    )

    with HttpClient() as http:
        client = MOEXClient(http=http, raw_dir=tmp_path)
        result = load_catalog(client, board="TQBR")

    assert isinstance(result, CatalogResult)
    assert set(result.common_shares["SECID"]) == {"SBER", "LKOH"}
    assert "SBERP" not in result.common_shares["SECID"].tolist()
    assert result.raw_paths


@responses.activate
def test_load_catalog_paginates(tmp_path):
    # Один кандидат на первой странице -> сбор продолжается на start=1.
    responses.get(
        CATALOG_URL,
        json=_catalog_payload([_catalog_row("SBER", isin="RU_a")]),
        status=200,
        match=[responses.matchers.query_param_matcher({"start": 0}, strict_match=False)],
    )
    responses.get(
        CATALOG_URL,
        json=_catalog_payload([_catalog_row("LKOH", isin="RU_b")]),
        status=200,
        match=[responses.matchers.query_param_matcher({"start": 1}, strict_match=False)],
    )
    responses.get(
        CATALOG_URL,
        json=_catalog_payload([]),
        status=200,
        match=[responses.matchers.query_param_matcher({"start": 2}, strict_match=False)],
    )

    with HttpClient() as http:
        client = MOEXClient(http=http, raw_dir=tmp_path)
        result = load_catalog(client, board="TQBR")

    assert set(result.common_shares["SECID"]) == {"SBER", "LKOH"}
    assert len(result.raw_paths) == 3


def test_load_catalog_empty_returns_empty(tmp_path):
    # Без сети: пустая таблица на входе фильтрации.
    from equity_factor_study.pipelines.catalog import _filter_common_shares

    empty = pd.DataFrame()
    filtered = _filter_common_shares(empty, "1")
    assert filtered.empty


def test_filter_common_shares_drops_missing_secid():
    from equity_factor_study.pipelines.catalog import _filter_common_shares

    table = pd.DataFrame({"SECID": ["SBER", "LKOH", None, ""], "SECTYPE": ["1", "1", "1", "1"]})
    filtered = _filter_common_shares(table, "1")
    assert list(filtered["SECID"]) == ["SBER", "LKOH"]


@responses.activate
def test_load_catalog_resolves_semantic_security_type(tmp_path):
    """Метка 'common_share' из конфигурации приводится к коду ISS '1'."""
    from equity_factor_study.pipelines.catalog import load_catalog, resolve_security_type

    assert resolve_security_type("common_share") == "1"
    assert resolve_security_type("1") == "1"
    with pytest.raises(ValueError, match="Неизвестный тип бумаги"):
        resolve_security_type("preferred_share")

    # load_catalog с меткой должен отфильтровать так же, как с кодом.
    responses.get(
        CATALOG_URL,
        json=_catalog_payload([_catalog_row("SBER", sectype="1"), _catalog_row("SBERP", sectype="2")]),
        status=200,
        match=[responses.matchers.query_param_matcher({"start": 0}, strict_match=False)],
    )
    responses.get(
        CATALOG_URL,
        json=_catalog_payload([]),
        status=200,
        match=[responses.matchers.query_param_matcher({"start": 2}, strict_match=False)],
    )
    with HttpClient() as http:
        client = MOEXClient(http=http, raw_dir=tmp_path)
        result = load_catalog(client, board="TQBR", security_type="common_share")
    assert list(result.common_shares["SECID"]) == ["SBER"]


# --- сбор оборота и выборка ---------------------------------------------------


@responses.activate
def test_build_universe_selects_top_by_turnover(tmp_path):
    # Каждый кандидат -> одна страница истории с оборотами.
    candidates = ["AAA", "BBB", "CCC"]
    turnover_map = {"AAA": 300.0, "BBB": 500.0, "CCC": 100.0}

    for candidate in candidates:
        responses.get(
            f"{HISTORY_URL}/{candidate}.json",
            json=_history_payload(candidate, turnover_map[candidate]),
            status=200,
            match=[responses.matchers.query_param_matcher({"start": 0}, strict_match=False)],
        )

    with HttpClient() as http:
        client = MOEXClient(http=http, raw_dir=tmp_path)
        result = build_universe(
            client=client,
            candidates=candidates,
            board="TQBR",
            year=2014,
            target_size=2,
        )

    # Отобраны два наибольших: BBB и AAA.
    assert result.selected == ["BBB", "AAA"]
    assert set(result.reasons["secid"]) == {"CCC"}
    assert result.reasons.iloc[0]["reason"] == "not_selected_low_turnover"


def test_build_universe_deterministic_ranking():
    from equity_factor_study.pipelines.universe import _rank_by_turnover

    turnover = pd.DataFrame(
        {"secid": ["A", "B", "C"], "turnover": [10.0, 10.0, 5.0]}
    )
    ranked = _rank_by_turnover(turnover)
    # равенство по обороту разрешается алфавитно => детерминизм.
    assert list(ranked["secid"]) == ["A", "B", "C"]


def test_build_universe_empty_candidates_raises():
    with pytest.raises(UniverseError):
        build_universe(
            client=None,  # type: ignore[arg-type]
            candidates=[],
            board="TQBR",
            year=2014,
            target_size=10,
        )


# --- lock-файл и исключения ---------------------------------------------------


def test_write_and_load_lock_file(tmp_path):
    lock_path = tmp_path / "configs" / "universe.lock.toml"
    write_lock_file(
        lock_path=lock_path,
        config_hash="hash123",
        version="0.1",
        board="TQBR",
        year=2014,
        selected=["SBER", "LKOH"],
        exclusions=pd.DataFrame({"secid": ["GAZP"]}),
    )
    assert load_lock_file(lock_path=lock_path) == ["SBER", "LKOH"]


def test_load_lock_file_missing_raises(tmp_path):
    with pytest.raises(UniverseError, match="не найден"):
        load_lock_file(lock_path=tmp_path / "universe.lock.toml")


def test_write_exclusions_records_reasons(tmp_path):
    path = tmp_path / "exclusions.csv"
    reasons = pd.DataFrame(
        {"secid": ["GAZP"], "reason": ["not_selected_low_turnover"], "turnover": [50.0], "rank": [11]}
    )
    write_exclusions(exclusions_path=path, reasons=reasons)
    content = path.read_text(encoding="utf-8")
    assert "not_selected_low_turnover" in content
    assert "GAZP" in content


@responses.activate
def test_build_and_write_universe_end_to_end(tmp_path):
    candidates = ["A", "B", "C"]
    turnover_map = {"A": 150.0, "B": 400.0, "C": 200.0}
    for candidate in candidates:
        responses.get(
            f"{HISTORY_URL}/{candidate}.json",
            json=_history_payload(candidate, turnover_map[candidate]),
            status=200,
            match=[responses.matchers.query_param_matcher({"start": 0}, strict_match=False)],
        )

    catalog_df = pd.DataFrame({"SECID": candidates, "SECTYPE": ["1"] * 3})

    lock_path = tmp_path / "configs" / "universe.lock.toml"
    exclusions_path = tmp_path / "exclusions.csv"

    with HttpClient() as http:
        client = MOEXClient(http=http, raw_dir=tmp_path)
        result = build_and_write_universe(
            client=client,
            lock_path=lock_path,
            exclusions_path=exclusions_path,
            catalog_table=catalog_df,
            config_hash="cfg",
            version="0.1",
            board="TQBR",
            year=2014,
            target_size=2,
        )

    assert result.selected == ["B", "C"]
    assert load_lock_file(lock_path=lock_path) == ["B", "C"]
    assert exclusions_path.exists()


@responses.activate
def test_build_and_write_universe_empty_catalog_raises(tmp_path):
    from equity_factor_study.pipelines.universe import build_and_write_universe

    with HttpClient() as http:
        client = MOEXClient(http=http, raw_dir=tmp_path)
        with pytest.raises(UniverseError, match="Справочник пуст"):
            build_and_write_universe(
                client=client,
                lock_path=tmp_path / "lock.toml",
                exclusions_path=tmp_path / "excl.csv",
                catalog_table=pd.DataFrame(),
                config_hash="cfg",
                version="0.1",
                board="TQBR",
                year=2014,
                target_size=10,
            )

def test_write_lock_file_records_turnover_and_formation(tmp_path):
    from datetime import date

    lock_path = tmp_path / "configs" / "universe.lock.toml"
    turnover = pd.DataFrame({"secid": ["SBER", "LKOH"], "turnover": [300.0, 200.0]})
    write_lock_file(
        lock_path=lock_path,
        config_hash="hash123",
        version="0.1",
        board="TQBR",
        year=2014,
        selected=["SBER", "LKOH"],
        turnover=turnover,
        formation_date=date(2024, 1, 1),
        exclusions_relative="universe_exclusions.csv",
    )
    with lock_path.open("rb") as file:
        data = tomllib.load(file)
    assert "instruments" in data
    tickers = {entry["secid"]: entry["turnover"] for entry in data["instruments"]}
    assert tickers == {"SBER": 300.0, "LKOH": 200.0}
    assert data["universe"]["formation_date"] == "2024-01-01"
    assert data["universe"]["exclusions_file"] == "universe_exclusions.csv"


@responses.activate
def test_build_and_write_universe_existing_lock_raises_unless_forced(tmp_path):
    from equity_factor_study.pipelines.universe import build_and_write_universe

    candidates = ["A", "B", "C"]
    turnover_map = {"A": 150.0, "B": 400.0, "C": 200.0}
    for candidate in candidates:
        responses.get(
            f"{HISTORY_URL}/{candidate}.json",
            json=_history_payload(candidate, turnover_map[candidate]),
            status=200,
            match=[responses.matchers.query_param_matcher({"start": 0}, strict_match=False)],
        )

    catalog_df = pd.DataFrame({"SECID": candidates, "SECTYPE": ["1"] * 3})
    lock_path = tmp_path / "configs" / "universe.lock.toml"
    exclusions_path = tmp_path / "exclusions.csv"

    with HttpClient() as http:
        client = MOEXClient(http=http, raw_dir=tmp_path / "run1")
        # Первый прогон создаёт замок.
        build_and_write_universe(
            client=client, lock_path=lock_path, exclusions_path=exclusions_path,
            catalog_table=catalog_df, config_hash="cfg", version="0.1",
            board="TQBR", year=2014, target_size=2,
        )
        original = lock_path.read_bytes()
        # Повторный прогон без подтверждения завершается ошибкой и не перезаписывает.
        with pytest.raises(UniverseError, match="перезапись"):
            build_and_write_universe(
                client=client, lock_path=lock_path, exclusions_path=exclusions_path,
                catalog_table=catalog_df, config_hash="cfg", version="0.1",
                board="TQBR", year=2014, target_size=2,
            )
        assert lock_path.read_bytes() == original
    # Явное подтверждение пересборки (force) разрешает перезапись.
    with HttpClient() as http:
        client = MOEXClient(http=http, raw_dir=tmp_path / "run2")
        result = build_and_write_universe(
            client=client, lock_path=lock_path, exclusions_path=exclusions_path,
            catalog_table=catalog_df, config_hash="cfg", version="0.1",
            board="TQBR", year=2014, target_size=2, force=True,
        )
        assert result.selected == ["B", "C"]


def test_append_manual_exclusions_validates_reasons(tmp_path):
    from equity_factor_study.pipelines.universe import (
        ALLOWED_MANUAL_REASONS,
        UniverseError,
        append_manual_exclusions,
    )

    path = tmp_path / "exclusions.csv"
    # Допустимая причина (качество данных).
    append_manual_exclusions(
        exclusions_path=path,
        rows=[{"secid": "GAZP", "reason": "data_gap_unexplained", "turnover": 50.0, "rank": 11}],
    )
    content = path.read_text(encoding="utf-8")
    assert "data_gap_unexplained" in content
    assert "GAZP" in content

    # Недопустимое основание (связано с доходностью) отклоняется.
    with pytest.raises(UniverseError, match="Недопустимое основание"):
        append_manual_exclusions(
            exclusions_path=path,
            rows=[{"secid": "AAA", "reason": "unusual_issuer_expected_low_return"}],
        )
    assert "AAA" not in path.read_text(encoding="utf-8")
    assert "data_gap_unexplained" in ALLOWED_MANUAL_REASONS
