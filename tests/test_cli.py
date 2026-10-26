"""Тесты CLI этапа 1: справка, код возврата и обработка ошибок."""

from datetime import date
from pathlib import Path

import pytest
import responses

from equity_factor_study.cli import build_parser, main
from equity_factor_study.config import load_config

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "study.toml"

CBR_URL = "https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx"

CATALOG_URL = (
    "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
)
HISTORY_URL = (
    "https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities"
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


def test_help_lists_commands(capsys):
    """Справка выводит список подкоманд."""
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    for command in (
        "check-config", "probe", "catalog", "universe", "prices",
        "key-rate", "audit", "panel",
    ):
        assert command in captured.out


def test_check_config_exit_zero(capsys):
    code = main(["check-config"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Конфигурация корректна" in captured.out
    assert "2018-06-01" in captured.out  # первый целевой месяц


@pytest.mark.parametrize("command", ["probe", "audit", "panel"])
def test_stub_commands_return_not_implemented(command, capsys):
    """Команды-заглушки выводят сообщение и возвращают код 2."""
    code = main([command])
    assert code == 2
    captured = capsys.readouterr()
    assert "ещё не реализован" in captured.err


@responses.activate
def test_catalog_command_uses_mock(tmp_path, capsys):
    """catalog: сеть мокается, вывод пишется во временный каталог, а не в data/raw."""
    responses.get(
        CATALOG_URL,
        json=_catalog_payload([
            _catalog_row("SBER", isin="RU_a"),
            _catalog_row("LKOH", isin="RU_b"),
        ]),
        status=200,
        match=[responses.matchers.query_param_matcher({"start": 0}, strict_match=False)],
    )
    responses.get(
        CATALOG_URL,
        json=_catalog_payload([]),
        status=200,
        match=[responses.matchers.query_param_matcher({"start": 2}, strict_match=False)],
    )

    code = main(["--config", str(DEFAULT_CONFIG), "--raw-dir", str(tmp_path), "catalog"])
    assert code == 0
    captured = capsys.readouterr()
    assert "строк" in captured.out
    assert "Обыкновенных акций: 2" in captured.out


@responses.activate
def test_universe_command_uses_mock(tmp_path):
    """universe: сеть мокается, замок пишется во временный каталог, а не в configs/."""
    candidates = ["A", "B", "C"]
    turnover_map = {"A": 150.0, "B": 400.0, "C": 200.0}

    responses.get(
        CATALOG_URL,
        json=_catalog_payload([_catalog_row(c, isin=f"RU_{c}") for c in candidates]),
        status=200,
        match=[responses.matchers.query_param_matcher({"start": 0}, strict_match=False)],
    )
    responses.get(
        CATALOG_URL,
        json=_catalog_payload([]),
        status=200,
        match=[responses.matchers.query_param_matcher({"start": 3}, strict_match=False)],
    )
    for candidate in candidates:
        responses.get(
            f"{HISTORY_URL}/{candidate}.json",
            json=_history_payload(candidate, turnover_map[candidate]),
            status=200,
            match=[responses.matchers.query_param_matcher({"start": 0}, strict_match=False)],
        )

    # Артефакты этапа 4 пишутся рядом с конфигурацией: копируем её во временный
    # каталог, чтобы не засорять реальный configs/.
    tmp_config_dir = tmp_path / "configs"
    tmp_config_dir.mkdir()
    tmp_config = tmp_config_dir / "study.toml"
    tmp_config.write_text(DEFAULT_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")

    code = main(["--config", str(tmp_config), "--raw-dir", str(tmp_path), "universe"])
    assert code == 0
    assert (tmp_config_dir / "universe.lock.toml").exists()
    assert (tmp_config_dir / "universe_exclusions.csv").exists()


def test_missing_config_returns_one(capsys, tmp_path):
    code = main(["--config", str(tmp_path / "missing.toml"), "check-config"])
    assert code == 1
    captured = capsys.readouterr()
    assert "не найден" in captured.err


def test_no_command_returns_usage_error():
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 2  # argparse: обязательная подкоманда


def test_control_dates_from_project_config():
    """Контрольные даты Б.6 подтверждаются объектом конфигурации проекта."""
    config = load_config(PROJECT_ROOT / "configs" / "study.toml")
    assert config.first_observation_month == date(2014, 5, 1)
    assert config.first_target_month == date(2018, 6, 1)
    assert config.period_length_months == 153


def _setup_tmp_config(tmp_path):
    """Копия конфигурации + замок состава с инструментами A, B."""
    tmp_config_dir = tmp_path / "configs"
    tmp_config_dir.mkdir()
    tmp_config = tmp_config_dir / "study.toml"
    tmp_config.write_text(DEFAULT_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_config_dir / "universe.lock.toml").write_text(
        "[universe]\nsecids = [\"A\", \"B\"]\n", encoding="utf-8"
    )
    return tmp_config


@responses.activate
def test_prices_command_uses_mock(tmp_path, capsys):
    """prices: сеть мокается, дневная таблица пишется во временный каталог."""
    tmp_config = _setup_tmp_config(tmp_path)
    def _payload(secid, rows):
        return {
            "history": {"columns": ["SECID", "BOARDID", "TRADEDATE", "CLOSE", "VALUE"], "data": rows},
            "history.cursor": {"columns": ["INDEX", "TOTAL", "PAGESIZE"], "data": [[0, len(rows), 100]]},
        }
    for secid in ["A", "B"]:
        responses.get(
            f"{HISTORY_URL}/{secid}.json",
            json=_payload(secid, [[secid, "TQBR", "2014-01-06", 10.0, 100.0]]),
            status=200,
            match=[responses.matchers.query_param_matcher({"start": 0}, strict_match=False)],
        )
        responses.get(
            f"{HISTORY_URL}/{secid}.json",
            json=_payload(secid, []),
            status=200,
            match=[responses.matchers.query_param_matcher({"start": 1}, strict_match=False)],
        )

    code = main(["--config", str(tmp_config), "--raw-dir", str(tmp_path), "--interim-dir", str(tmp_path), "prices"])
    assert code == 0
    assert (tmp_path / "daily_prices.csv").exists()
    assert "Загружено инструментов: 2" in capsys.readouterr().out


def test_prices_command_missing_lock_returns_one(tmp_path, capsys):
    """prices без файла-замка состава завершается кодом 1."""
    tmp_config_dir = tmp_path / "configs"
    tmp_config_dir.mkdir()
    tmp_config = tmp_config_dir / "study.toml"
    tmp_config.write_text(DEFAULT_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")

    code = main(["--config", str(tmp_config), "--interim-dir", str(tmp_path), "prices"])
    assert code == 1


@responses.activate
def test_key_rate_command_uses_mock(tmp_path, capsys):
    """key-rate: сеть мокается, ряд пишется во временный каталог."""
    tmp_config = _setup_tmp_config(tmp_path)
    responses.post(
        CBR_URL,
        body=(Path(__file__).resolve().parent / "fixtures" / "cbr_keyrate_2014-01.xml").read_bytes(),
        status=200,
        content_type="text/xml",
    )

    code = main(["--config", str(tmp_config), "--raw-dir", str(tmp_path), "--interim-dir", str(tmp_path), "key-rate"])
    assert code == 0
    assert (tmp_path / "key_rate.csv").exists()
    assert "Строк в ряду ставки" in capsys.readouterr().out