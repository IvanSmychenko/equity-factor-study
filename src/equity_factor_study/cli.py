"""Единая точка входа командной строки для исследования.

Подкоманды соответствуют этапам сбора и построения панели. На этапе 1 часть
команд являются заглушками, но обязаны выводить понятное сообщение и
завершаться осмысленным кодом возврата.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from equity_factor_study.config import ConfigError, StudyConfig, load_config

DEFAULT_CONFIG_PATH = Path("configs/study.toml")

# Код возврата для команд-заглушек, ещё не реализованных в текущем этапе.
_EXIT_NOT_IMPLEMENTED = 2


def _load_config(arguments: argparse.Namespace) -> StudyConfig:
    return load_config(arguments.config)


def _cmd_check_config(arguments: argparse.Namespace) -> int:
    """Проверить конфигурацию и вывести её сводку."""
    config = _load_config(arguments)
    print(f"Конфигурация корректна: {arguments.config.resolve()}")
    print(f"  Версия протокола: {config.protocol.version}")
    print(f"  Период наблюдения: {config.period.observation_start} — {config.period.observation_end}")
    print(f"  Первый месяц с полным набором признаков: {config.first_observation_month}")
    print(f"  Первый целевой месяц: {config.first_target_month}")
    print(f"  Целевой размер выборки: {config.universe.target_size}")
    print(f"  Окно обучения (мес.): {config.model.training_window_months}")
    return 0


def _unsupported(stage_name: str) -> Callable[[argparse.Namespace], int]:
    """Создать команду-заглушку для будущего этапа."""

    def _run(arguments: argparse.Namespace) -> int:
        config = _load_config(arguments)
        print(
            f"Этап «{stage_name}» ещё не реализован. "
            f"Конфигурация загружена из {arguments.config.resolve()} "
            f"(протокол {config.protocol.version}).",
            file=sys.stderr,
        )
        return _EXIT_NOT_IMPLEMENTED

    _run.__name__ = f"cmd_{stage_name}"
    return _run


def build_parser() -> argparse.ArgumentParser:
    """Построить парсер аргументов командной строки."""
    parser = argparse.ArgumentParser(
        prog="equity-factor-study",
        description="Сбор и подготовка данных для исследования доходности российских акций.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Путь к файлу конфигурации (по умолчанию {DEFAULT_CONFIG_PATH}).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")
    subparsers.required = True

    check_parser = subparsers.add_parser(
        "check-config",
        help="Проверить корректность конфигурации и вывести сводку.",
    )
    check_parser.set_defaults(handler=_cmd_check_config)

    # Коллекция данных источников.
    probe_parser = subparsers.add_parser(
        "probe",
        help="Пробный запрос к источнику (заглушка этапа сбора).",
    )
    probe_parser.set_defaults(handler=_unsupported("probe"))

    catalog_parser = subparsers.add_parser(
        "catalog",
        help="Справочники инструментов и режимов торгов (заглушка).",
    )
    catalog_parser.set_defaults(handler=_unsupported("catalog"))

    universe_parser = subparsers.add_parser(
        "universe",
        help="Построение исторической совокупности (заглушка).",
    )
    universe_parser.set_defaults(handler=_unsupported("universe"))

    prices_parser = subparsers.add_parser(
        "prices",
        help="Загрузка дневных цен по выборке (заглушка).",
    )
    prices_parser.set_defaults(handler=_unsupported("prices"))

    key_rate_parser = subparsers.add_parser(
        "key-rate",
        help="Загрузка дневного ряда ключевой ставки (заглушка).",
    )
    key_rate_parser.set_defaults(handler=_unsupported("key-rate"))

    audit_parser = subparsers.add_parser(
        "audit",
        help="Диагностика качества и корпоративные события (заглушка).",
    )
    audit_parser.set_defaults(handler=_unsupported("audit"))

    panel_parser = subparsers.add_parser(
        "panel",
        help="Построение месячной панели и манифеста (заглушка).",
    )
    panel_parser.set_defaults(handler=_unsupported("panel"))

    return parser


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI. Возвращает код завершения."""
    parser = build_parser()
    arguments = parser.parse_args(argv)

    handler: Callable[[argparse.Namespace], int] = arguments.handler
    try:
        return handler(arguments)
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nПрервано пользователем.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())