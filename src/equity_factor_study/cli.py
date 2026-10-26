"""Единая точка входа командной строки для исследования.

Подкоманды соответствуют этапам сбора и построения панели. На этапе 1 часть
команд являются заглушками, но обязаны выводить понятное сообщение и
завершаться осмысленным кодом возврата.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Callable
from pathlib import Path

from equity_factor_study.clients.moex import MOEXClient
from equity_factor_study.config import ConfigError, StudyConfig, load_config
from equity_factor_study.http import HttpClient
from equity_factor_study.storage import run_dir

DEFAULT_CONFIG_PATH = Path("configs/study.toml")
DEFAULT_RAW_DIR = Path("data") / "raw"
DEFAULT_INTERIM_DIR = Path("data") / "interim"

# Код возврата для команд-заглушек, ещё не реализованных в текущем этапе.
_EXIT_NOT_IMPLEMENTED = 2


def _load_config(arguments: argparse.Namespace) -> StudyConfig:
    return load_config(arguments.config)


def _config_hash(config_path: Path) -> str:
    """Вычислить SHA-256 содержимого файла конфигурации."""
    digest = hashlib.sha256()
    digest.update(config_path.read_bytes())
    return digest.hexdigest()


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


def _cmd_catalog(arguments: argparse.Namespace) -> int:
    """Собрать справочник обыкновенных акций на основном режиме."""
    config = _load_config(arguments)
    from equity_factor_study.pipelines.catalog import load_catalog

    with HttpClient() as http:
        client = MOEXClient(http=http, raw_dir=run_dir(base_dir=arguments.raw_dir, source="moex"))
        result = load_catalog(
            client=client,
            board=config.universe.selection_board,
            security_type=config.universe.security_type,
        )

    print(f"Справочник режима {config.universe.selection_board}: {len(result.table)} строк")
    print(f"Обыкновенных акций: {len(result.common_shares)}")
    return 0


def _cmd_universe(arguments: argparse.Namespace) -> int:
    """Построить историческую совокупность и записать файл-замок."""
    config = _load_config(arguments)
    from equity_factor_study.pipelines.catalog import load_catalog
    from equity_factor_study.pipelines.universe import build_and_write_universe

    # Артефакты этапа 4 (замок состава и перечень исключений) размещаются
    # рядом с файлом конфигурации (по ТЗ — configs/universe.lock.toml),
    # а не внутри пер-прогонной выгрузки raw-данных.
    config_dir = arguments.config.resolve().parent

    with HttpClient() as http:
        client = MOEXClient(http=http, raw_dir=run_dir(base_dir=arguments.raw_dir, source="moex"))
        catalog = load_catalog(
            client=client,
            board=config.universe.selection_board,
            security_type=config.universe.security_type,
        )
        result = build_and_write_universe(
            client=client,
            lock_path=config_dir / "universe.lock.toml",
            exclusions_path=config_dir / "universe_exclusions.csv",
            catalog_table=catalog.common_shares,
            config_hash=_config_hash(arguments.config),
            version=config.protocol.version,
            board=config.universe.selection_board,
            year=config.universe.turnover_year,
            target_size=config.universe.target_size,
        )

    print(f"Отобрано инструментов: {len(result.selected)}: {', '.join(result.selected)}")
    print(f"Исключено кандидатов: {len(result.reasons)}")
    print(f"Замок состава: {config_dir / 'universe.lock.toml'}")
    return 0


def _load_lock_secids(arguments: argparse.Namespace) -> list[str]:
    """Прочитать состав выборки из файла-замка рядом с конфигурацией."""
    from equity_factor_study.pipelines.universe import load_lock_file

    lock_path = arguments.config.resolve().parent / "universe.lock.toml"
    return load_lock_file(lock_path=lock_path)


def _cmd_prices(arguments: argparse.Namespace) -> int:
    """Загрузить дневные цены по выборке и нормализовать в дневную таблицу."""
    config = _load_config(arguments)
    from equity_factor_study.pipelines.prices import collect_prices, write_daily_prices
    from equity_factor_study.pipelines.universe import UniverseError

    try:
        secids = _load_lock_secids(arguments)
    except UniverseError as exc:
        print(f"Ошибка выборки: {exc}", file=sys.stderr)
        return 1
    if not secids:
        print("Пустая выборка: нет инструментов для загрузки цен.", file=sys.stderr)
        return 1

    with HttpClient() as http:
        client = MOEXClient(http=http, raw_dir=run_dir(base_dir=arguments.raw_dir, source="moex"))
        result = collect_prices(
            client=client,
            secids=secids,
            board=config.universe.selection_board,
            from_date=config.period.download_start,
            to_date=config.period.observation_end,
        )

    out_path = write_daily_prices(result.table, arguments.interim_dir / "daily_prices.csv")
    print(f"Загружено инструментов: {len(result.summaries)}")
    print(f"Строк в дневной таблице: {result.rows}")
    print(f"Сохранено: {out_path.resolve()}")
    for summary in result.summaries:
        status = "OK" if summary.complete else f"пропусков: {summary.missing_trading_days}"
        print(
            f"  {summary.secid}: {summary.rows} строк, "
            f"{summary.date_from}..{summary.date_to}, {status}"
        )
    return 0


def _cmd_key_rate(arguments: argparse.Namespace) -> int:
    """Загрузить дневной ряд ключевой ставки и нормализовать его."""
    config = _load_config(arguments)
    from equity_factor_study.clients.cbr import CBRClient
    from equity_factor_study.pipelines.key_rate import collect_key_rate, write_daily_key_rate

    with HttpClient() as http:
        client = CBRClient(http=http, raw_dir=run_dir(base_dir=arguments.raw_dir, source="cbr"))
        result = collect_key_rate(
            client=client,
            from_date=config.period.observation_start,
            to_date=config.period.observation_end,
        )

    out_path = write_daily_key_rate(result.table, arguments.interim_dir / "key_rate.csv")
    summary = result.summary
    print(f"Строк в ряду ставки: {summary.rows}")
    print(f"Период: {summary.date_from}..{summary.date_to}")
    print(f"Разрывов покрытия: {summary.gaps}")
    status = "OK" if summary.complete else "НЕПОЛНОЕ ПОКРЫТИЕ"
    print(f"Статус: {status}")
    print(f"Сохранено: {out_path.resolve()}")
    return 0


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
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Каталог выгрузки исходных данных (по умолчанию {DEFAULT_RAW_DIR}).",
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=DEFAULT_INTERIM_DIR,
        help=f"Каталог промежуточных таблиц (по умолчанию {DEFAULT_INTERIM_DIR}).",
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
        help="Справочники инструментов и режимов торгов.",
    )
    catalog_parser.set_defaults(handler=_cmd_catalog)

    universe_parser = subparsers.add_parser(
        "universe",
        help="Построение исторической совокупности и запись файла-замка.",
    )
    universe_parser.set_defaults(handler=_cmd_universe)

    prices_parser = subparsers.add_parser(
        "prices",
        help="Загрузка дневных цен по выборке.",
    )
    prices_parser.set_defaults(handler=_cmd_prices)

    key_rate_parser = subparsers.add_parser(
        "key-rate",
        help="Загрузка дневного ряда ключевой ставки.",
    )
    key_rate_parser.set_defaults(handler=_cmd_key_rate)

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