"""Тесты CLI этапа 1: справка, код возврата и обработка ошибок."""

from datetime import date

import pytest

from equity_factor_study.cli import build_parser, main
from equity_factor_study.config import load_config

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_help_lists_commands(capsys):
    """Справка выводит список подкоманд."""
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    for command in ("check-config", "probe", "catalog", "universe", "prices", "key-rate", "audit", "panel"):
        assert command in captured.out


def test_check_config_exit_zero(capsys):
    code = main(["check-config"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Конфигурация корректна" in captured.out
    assert "2018-06-01" in captured.out  # первый целевой месяц


@pytest.mark.parametrize("command", ["probe", "catalog", "universe", "prices", "key-rate", "audit", "panel"])
def test_stub_commands_return_not_implemented(command, capsys):
    """Команды-заглушки выводят сообщение и возвращают код 2."""
    code = main([command])
    assert code == 2
    captured = capsys.readouterr()
    assert "ещё не реализован" in captured.err


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