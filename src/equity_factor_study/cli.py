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

import pandas as pd

from equity_factor_study.clients.moex import MOEXClient
from equity_factor_study.config import ConfigError, StudyConfig, load_config
from equity_factor_study.http_client import HttpClient
from equity_factor_study.storage import run_dir

DEFAULT_CONFIG_PATH = Path("configs/study.toml")
DEFAULT_RAW_DIR = Path("data") / "raw"
DEFAULT_INTERIM_DIR = Path("data") / "interim"
DEFAULT_REPORT_DIR = Path("data") / "reports"
DEFAULT_PROCESSED_DIR = Path("data") / "processed"

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
            force=bool(getattr(arguments, "force", False)),
        )

    print(f"Отобрано инструментов: {len(result.selected)}: {', '.join(result.selected)}")
    print(f"Исключено кандидатов: {len(result.reasons)}")
    print(f"Замок состава: {config_dir / 'universe.lock.toml'}")
    return 0


def _cmd_review(arguments: argparse.Namespace) -> int:
    """Внести результат ручной проверки состава в перечень исключений.

    Ручное исключение допускается только по причинам качества данных; любое
    иное основание (например, ожидаемая доходность эмитента) отклоняется.
    """
    from equity_factor_study.pipelines.universe import (
        ALLOWED_MANUAL_REASONS,
        UniverseError,
        append_manual_exclusions,
    )

    rows = []
    for secid, reason in arguments.exclude:
        rows.append({"secid": secid, "reason": reason})
    if not rows:
        print("Не указаны исключения. Используйте --exclude SECID:PRICHINA.")
        print("Допустимые причины: " + ", ".join(sorted(ALLOWED_MANUAL_REASONS)))
        return 1

    exclusions_path = arguments.config.resolve().parent / "universe_exclusions.csv"
    try:
        append_manual_exclusions(exclusions_path=exclusions_path, rows=rows)
    except UniverseError as exc:
        print(f"Ошибка: {exc}")
        return 2
    print(f"Добавлены исключения в: {exclusions_path}")
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


def _cmd_audit(arguments: argparse.Namespace) -> int:
    """Диагностика качества дневных цен и корпоративные события."""
    from equity_factor_study.pipelines.audit import (
        load_corporate_events,
        run_audit,
        write_events,
        write_findings,
        write_summary,
    )

    prices_path = arguments.interim_dir / "daily_prices.csv"
    if not prices_path.exists():
        print(f"Не найдена дневная таблица цен: {prices_path.resolve()}", file=sys.stderr)
        return 1

    prices = _read_prices(prices_path)

    # Корпоративные события — обязательный вход диагностики (шаг A1):
    # отсутствие файла не должно трактоваться как «событий нет».
    events_path_in = arguments.interim_dir / "corporate_actions.csv"
    if not events_path_in.exists():
        print(
            f"Не найдена таблица корпоративных событий: {events_path_in.resolve()}",
            file=sys.stderr,
        )
        return 1
    events = load_corporate_events(events_path_in)
    result = run_audit(prices, events=events)

    findings_path = write_findings(result.findings, arguments.report_dir / "findings.csv")
    events_path = write_events(result.events, arguments.report_dir / "corporate_events.csv")
    summary_path = write_summary(result.summary, arguments.report_dir / "quality_summary.csv")

    print(f"Найдено замечаний: {len(result.findings)}")
    print(f"Корпоративных событий: {len(result.events)}")
    if not result.summary.empty:
        for _, row in result.summary.iterrows():
            print(f"  {row['type']}: {row['count']}")
    print(f"Замечания: {findings_path.resolve()}")
    print(f"События: {events_path.resolve()}")
    print(f"Сводка: {summary_path.resolve()}")
    return 0


def _cmd_collect_events(arguments: argparse.Namespace) -> int:
    """Зафиксировать корпоративное событие в таблице событий (шаг A2).

    Полуручной сбор события с обязательной фиксацией ``source`` (происхождение)
    и ``retrieved_at`` (дата получения). Используется, когда машинный маршрут
    недоступен (сплит/консолидация ВТБ 15.07.2024, коэффициент вычислен по
    официальной истории цен).
    """
    from equity_factor_study.pipelines.audit import (
        load_corporate_events,
        record_corporate_event,
        save_corporate_events,
    )

    events_path = arguments.interim_dir / "corporate_actions.csv"
    # Существующие события сохраняем, новую строку добавляем.
    if events_path.exists():
        existing = load_corporate_events(events_path)
        rows = []
        for record in existing.to_dict("records"):
            rows.append(
                {
                    "secid": record["secid"],
                    "date": pd.Timestamp(record["date"]).date().isoformat(),
                    "event_type": record["event_type"],
                    "coef": record.get("coef"),
                    "source": record.get("source"),
                    "retrieved_at": record.get("retrieved_at"),
                }
            )
        new = record_corporate_event(
            arguments.secid,
            arguments.event_date,
            arguments.event_type,
            coef=arguments.coef,
            source=arguments.source,
            retrieved_at=arguments.retrieved_at,
        )
        rows.extend(new.to_dict("records"))
        events = pd.DataFrame(rows)[list(record_corporate_event("", "2000-01-01", "").columns)]
    else:
        events = record_corporate_event(
            arguments.secid,
            arguments.event_date,
            arguments.event_type,
            coef=arguments.coef,
            source=arguments.source,
            retrieved_at=arguments.retrieved_at,
        )

    save_corporate_events(events, events_path)
    print(f"Зафиксировано событие {arguments.secid} {arguments.event_date} "
          f"{arguments.event_type} (coef={arguments.coef}) -> {events_path.resolve()}")
    return 0


def _read_prices(path: Path) -> pd.DataFrame:
    """Прочитать нормализованную дневную таблицу цен."""
    table = pd.read_csv(path)
    table["trade_date"] = pd.to_datetime(table["trade_date"]).dt.date
    return table


def _read_key_rate(path: Path) -> pd.DataFrame:
    """Прочитать нормализованный ряд ключевой ставки."""
    table = pd.read_csv(path)
    table["date"] = pd.to_datetime(table["date"]).dt.date
    return table


def _sha256(path: Path) -> str:
    """Вычислить SHA-256 содержимого файла."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        digest.update(file.read())
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    """Путь относительно рабочего каталога, если возможно, иначе абсолютный."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path.resolve())


def _cmd_panel(arguments: argparse.Namespace) -> int:
    """Построить месячную панель и сформировать манифест."""
    config = _load_config(arguments)
    from equity_factor_study.pipelines.audit import load_corporate_events
    from equity_factor_study.pipelines.monthly import (
        build_manifest,
        build_monthly_panel,
    )

    prices_path = arguments.interim_dir / "daily_prices.csv"
    key_rate_path = arguments.interim_dir / "key_rate.csv"
    if not prices_path.exists():
        print(f"Не найдена дневная таблица цен: {prices_path.resolve()}", file=sys.stderr)
        return 1

    daily = _read_prices(prices_path)
    secids = sorted(set(daily["secid"]))
    key_rate = _read_key_rate(key_rate_path) if key_rate_path.exists() else None

    # Корпоративные события (обязательный вход при сборке панели, шаг A1):
    # приводят ряд к единому масштабу и участвуют в приёмке.
    events_path = arguments.interim_dir / "corporate_actions.csv"
    if not events_path.exists():
        print(
            f"Не найдена таблица корпоративных событий: {events_path.resolve()}",
            file=sys.stderr,
        )
        return 1
    events = load_corporate_events(events_path)

    panel = build_monthly_panel(
        daily,
        secids=secids,
        from_month=config.period.observation_start,
        to_month=config.period.observation_end,
        max_trading_days=config.month_end.max_trading_days_from_month_end,
        momentum_window=config.features.momentum_prev_3m,
        volatility_window=config.features.volatility_3m,
        key_rate=key_rate,
        events=events,
    )

    panel_path = arguments.processed_dir / "monthly_panel.csv"
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    out = panel.copy()
    out["month"] = out["month"].map(lambda d: d.isoformat() if hasattr(d, "isoformat") else d)
    out["price_date"] = out["price_date"].map(lambda d: d.isoformat() if hasattr(d, "isoformat") else d)
    out.to_csv(panel_path, index=False)

    source_files = [
        {"path": _display_path(prices_path), "sha256": _sha256(prices_path)},
    ]
    if key_rate_path.exists():
        source_files.append(
            {"path": _display_path(key_rate_path), "sha256": _sha256(key_rate_path)}
        )

    # Запись о применённых корректировках сплита в манифест (шаг A4).
    adjustments = []
    if not events.empty and "event_type" in events.columns:
        for _, row in events.iterrows():
            if row.get("event_type") == "split" and not pd.isna(row.get("coef")):
                adjustments.append(
                    {
                        "secid": row["secid"],
                        "date": pd.Timestamp(row["date"]).date().isoformat(),
                        "event_type": "split",
                        "coef": row["coef"],
                        "source": row.get("source"),
                    }
                )

    manifest = build_manifest(
        panel=panel,
        protocol_version=config.protocol.version,
        observation_start=config.period.observation_start,
        observation_end=config.period.observation_end,
        secids=secids,
        source_files=source_files,
        code_version="0.1.0",
        dependencies={},
        adjustments=adjustments or None,
    )

    manifest_path = arguments.processed_dir / "dataset_manifest.json"
    import json

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Панель: {panel_path.resolve()}, строк: {len(panel)}")
    print(f"Манифест: {manifest_path.resolve()}")
    return 0


def _cmd_acceptance(arguments: argparse.Namespace) -> int:
    """Приёмка этапа 8: прогнать проверки и сформировать отчёт."""
    from equity_factor_study.checks.acceptance import write_report

    config_dir = arguments.config.resolve().parent
    # Обязательные входы пайплайна: отсутствие любого делает приёмку возможной
    # только при выявлении расхождения (шаг A1), а не молчаливым пропуском.
    input_paths = [
        ("замок состава", config_dir / "universe.lock.toml"),
        ("дневные цены", arguments.interim_dir / "daily_prices.csv"),
        ("ключевая ставка", arguments.interim_dir / "key_rate.csv"),
        ("корпоративные события", arguments.interim_dir / "corporate_actions.csv"),
        ("месячная панель", arguments.processed_dir / "monthly_panel.csv"),
    ]
    report_path = write_report(arguments.report_dir, input_paths=input_paths)
    print(f"Отчёт приёмки: {report_path.resolve()}")
    return 0


def _cmd_evaluate(arguments: argparse.Namespace) -> int:
    """Оценить модель и полезность факторов на панели."""
    config = _load_config(arguments)
    from equity_factor_study.model.orchestrator import load_panel, run_model_evaluation

    panel_path = arguments.processed_dir / "monthly_panel.csv"
    if not panel_path.exists():
        print(f"Панель не найдена: {panel_path.resolve()}", file=sys.stderr)
        return 1

    feature_columns = _feature_columns_from_config(config)
    panel = load_panel(panel_path)

    report = run_model_evaluation(
        panel,
        feature_columns=feature_columns,
        training_window_months=config.model.training_window_months,
        refit_frequency=config.model.refit_frequency,
        first_target_month=config.first_target_month,
        development_target_end=config.period.development_target_end,
        last_month=config.period.observation_end,
    )

    _write_evaluation_report(arguments.report_dir, report)
    print(f"Оценка модели: прогнозов={report.full_metrics.forecast_count}")
    print(f"  MSE (оценка): {report.full_metrics.mse:.6f}")
    print(f"  MAE (оценка): {report.full_metrics.mae:.6f}")
    return 0


def _feature_columns_from_config(config) -> list[str]:
    """Список активных признаков по конфигурации (согласован со схемой панели)."""
    features = config.features
    columns = []
    if features.return_1m:
        columns.append("return_1m")
    if features.momentum_prev_3m:
        columns.append("momentum_prev_3m")
    if features.volatility_3m:
        columns.append("volatility_3m")
    if features.key_rate_pct:
        columns.append("key_rate_pct")
    if features.key_rate_change_pp:
        columns.append("key_rate_change_pp")
    return columns


def _write_evaluation_report(report_dir: Path, report) -> None:
    """Записать отчёт оценки модели в JSON."""
    import json

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_columns": report.feature_columns,
        "evaluation": {
            "forecast_count": report.full_metrics.forecast_count,
            "mse": report.full_metrics.mse,
            "mae": report.full_metrics.mae,
        },
        "factor_usefulness": report.factor_usefulness,
    }
    path = report_dir / "model_evaluation.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Отчёт оценки: {path.resolve()}")


def _cmd_selection(arguments: argparse.Namespace) -> int:
    """Запустить процедуру отбора состава факторов и сравнение моделей A/B/C."""
    import json

    config = _load_config(arguments)
    from equity_factor_study.model.orchestrator import load_panel
    from equity_factor_study.model.panel import build_observation_matrix
    from equity_factor_study.model.selection import run_selection_evaluation

    panel_path = arguments.processed_dir / "monthly_panel.csv"
    if not panel_path.exists():
        print(f"Панель не найдена: {panel_path.resolve()}", file=sys.stderr)
        return 1

    panel = load_panel(panel_path)
    matrix = build_observation_matrix(panel)
    frame = matrix.features.copy()
    frame["target_return_next_1m"] = matrix.target.to_numpy()
    frame["month"] = matrix.identity["month"].to_numpy()
    frame["secid"] = matrix.identity["secid"].to_numpy()

    report = run_selection_evaluation(
        frame,
        training_window_months=config.model.training_window_months,
        first_target_month=config.first_target_month,
        evaluation_target_start=config.period.evaluation_target_start,
        last_month=config.period.observation_end,
        selection_window_months=config.selection.selection_window_months,
        rebalance_frequency=config.selection.rebalance_frequency,
        per_security=config.model.per_security,
    )

    report_dir = Path(arguments.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_a": {
            "mse": report.model_a.mse,
            "mae": report.model_a.mae,
            "forecast_count": report.model_a.forecast_count,
        },
        "model_b": {
            "mse": report.model_b.mse,
            "mae": report.model_b.mae,
            "forecast_count": report.model_b.forecast_count,
        },
        "model_c": {
            "mse": report.model_c.mse,
            "mae": report.model_c.mae,
            "forecast_count": report.model_c.forecast_count,
        },
        "comparison_ba": report.comparison_ba,
        "comparison_ca": report.comparison_ca,
        "comparison_ba_win_rate": report.comparison_ba_win_rate,
        "rebalance_history": [
            {"rebalance_date": str(r["rebalance_date"]), "selected_groups": r["selected_groups"]}
            for r in report.rebalance_history.to_dict("records")
        ],
        "monthly_difference": [
            {"month": m.isoformat(), "value": float(v)}
            for m, v in (report.monthly_mse_c - report.monthly_mse_b).dropna().items()
        ],
    }
    path = report_dir / "selection_report.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Отчёт отбора: {path.resolve()}")
    print(f"  модель A: MSE={report.model_a.mse:.6f}, MAE={report.model_a.mae:.6f}")
    print(f"  модель B: MSE={report.model_b.mse:.6f}, MAE={report.model_b.mae:.6f}")
    print(f"  модель C: MSE={report.model_c.mse:.6f}, MAE={report.model_c.mae:.6f}")
    print(f"  C−B: {report.comparison_ba:.6f}   C−A: {report.comparison_ca:.6f}")
    return 0


def _cmd_sensitivity(arguments: argparse.Namespace) -> int:
    """Запустить проверки устойчивости процедуры отбора (раздел 4.3)."""
    import json

    config = _load_config(arguments)
    from equity_factor_study.model.orchestrator import load_panel
    from equity_factor_study.model.panel import build_observation_matrix
    from equity_factor_study.model.selection import run_selection_evaluation
    from equity_factor_study.model.sensitivity import (
        sensitivity_by_block,
        sensitivity_by_exclusion,
        sensitivity_by_extreme_months,
        sensitivity_by_metric,
        sensitivity_by_selection_window,
        sensitivity_by_window,
        sensitivity_summary,
    )

    panel_path = arguments.processed_dir / "monthly_panel.csv"
    if not panel_path.exists():
        print(f"Панель не найдена: {panel_path.resolve()}", file=sys.stderr)
        return 1

    panel = load_panel(panel_path)
    matrix = build_observation_matrix(panel)
    frame = matrix.features.copy()
    frame["target_return_next_1m"] = matrix.target.to_numpy()
    frame["month"] = matrix.identity["month"].to_numpy()
    frame["secid"] = matrix.identity["secid"].to_numpy()

    bcfg = config.bootstrap
    mcfg = config.model
    scfg = config.selection

    # Основная спецификация — эталон для сравнения с чувствительностью.
    base = run_selection_evaluation(
        frame,
        training_window_months=mcfg.training_window_months,
        first_target_month=config.first_target_month,
        evaluation_target_start=config.period.evaluation_target_start,
        last_month=config.period.observation_end,
        selection_window_months=scfg.selection_window_months,
        per_security=mcfg.per_security,
    )

    all_rows = []
    # 1. Длина обучающего окна.
    all_rows += sensitivity_by_window(
        frame, windows=[mcfg.training_window_months, *mcfg.window_sensitivity],
        first_target_month=config.first_target_month,
        evaluation_target_start=config.period.evaluation_target_start,
        last_month=config.period.observation_end,
        selection_window_months=scfg.selection_window_months,
        block_months=bcfg.block_months, replicates=bcfg.replicates,
        confidence=bcfg.confidence, seed=bcfg.seed,
    )
    # 2. Длина проверочного периода отбора.
    all_rows += sensitivity_by_selection_window(
        frame, training_window_months=mcfg.training_window_months,
        selection_windows=[scfg.selection_window_months, *scfg.selection_window_sensitivity],
        first_target_month=config.first_target_month,
        evaluation_target_start=config.period.evaluation_target_start,
        last_month=config.period.observation_end,
        block_months=bcfg.block_months, replicates=bcfg.replicates,
        confidence=bcfg.confidence, seed=bcfg.seed,
    )
    # 3. Длина блока бутстрэпа (без пересчёта моделей).
    all_rows += sensitivity_by_block(
        base, blocks=[bcfg.block_months, *bcfg.block_sensitivity],
        replicates=bcfg.replicates, confidence=bcfg.confidence, seed=bcfg.seed,
    )
    # 4. Состав выборки (исключение бумаг).
    all_rows += sensitivity_by_exclusion(
        frame, training_window_months=mcfg.training_window_months,
        first_target_month=config.first_target_month,
        evaluation_target_start=config.period.evaluation_target_start,
        last_month=config.period.observation_end,
        selection_window_months=scfg.selection_window_months,
        block_months=bcfg.block_months, replicates=bcfg.replicates,
        confidence=bcfg.confidence, seed=bcfg.seed,
    )
    # 5. Экстремальные месяцы.
    all_rows += sensitivity_by_extreme_months(
        base, block_months=bcfg.block_months, replicates=bcfg.replicates,
        confidence=bcfg.confidence, seed=bcfg.seed,
    )
    # 6. MAE как основная метрика.
    all_rows += sensitivity_by_metric(
        base, block_months=bcfg.block_months, replicates=bcfg.replicates,
        confidence=bcfg.confidence, seed=bcfg.seed,
    )

    table = sensitivity_summary(all_rows)
    report_dir = Path(arguments.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    out = table.to_dict("records")
    path = report_dir / "sensitivity_report.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Отчёт устойчивости: {path.resolve()}")
    for rec in out:
        ci = "содержит 0" if rec.get("ci_contains_zero") else "не содержит 0"
        print(f"  {rec['specification']:28s} C−B={rec['comparison_ba']:.6f} "
              f"CI=[{rec['lower']:.6f}, {rec['upper']:.6f}] {ci}")
    return 0


def _cmd_bootstrap(arguments: argparse.Namespace) -> int:
    """Бутстрэп-интервал разности MSE моделей B и C (парный блочный)."""
    import json

    config = _load_config(arguments)
    from equity_factor_study.model.bootstrap import paired_block_bootstrap

    selection_path = arguments.report_dir / "selection_report.json"
    if not selection_path.exists():
        print(f"Нет отчёта отбора: {selection_path.resolve()}", file=sys.stderr)
        return 1
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    # Ряд месячных разностей MSE (C−B): пересоздаём из истории пересмотров,
    # если она записана, иначе бутстрэп не выполним без месячного ряда.
    diff_rows = payload.get("monthly_difference")
    if not diff_rows:
        print(
            "Отчёт отбора не содержит ряда месячных разностей; запустите selection "
            "с сохранением ряда.",
            file=sys.stderr,
        )
        return 1
    monthly_diff = _series_from_records(diff_rows)

    block_months = config.bootstrap.block_months
    result = paired_block_bootstrap(
        monthly_diff,
        block_months=block_months,
        replicates=config.bootstrap.replicates,
        confidence=config.bootstrap.confidence,
        seed=config.bootstrap.seed,
    )

    report_dir = Path(arguments.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "point_estimate": result.point_estimate,
        "mean": result.mean,
        "lower": result.lower,
        "upper": result.upper,
        "replicates": result.replicates,
        "block_months": result.block_months,
        "confidence": result.confidence,
        "seed": result.seed,
        "n_observations": result.n_observations,
    }
    path = report_dir / "bootstrap_report.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Отчёт бутстрэпа: {path.resolve()}")
    print(f"  C−B тoч.оценка: {result.point_estimate:.8f}")
    print(f"  {int(result.confidence*100)}% интервал: [{result.lower:.8f}, {result.upper:.8f}]")
    print(f"  повторов: {result.replicates}, блок: {result.block_months} мес.")
    return 0


def _series_from_records(records: list[dict]) -> pd.Series:
    """Построить ряд из списка записей {month, value}."""
    keys = [r["month"] for r in records]
    vals = [r["value"] for r in records]
    return pd.Series(vals, index=pd.to_datetime(keys), dtype=float)


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
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help=f"Каталог диагностических отчётов (по умолчанию {DEFAULT_REPORT_DIR}).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help=f"Каталог итоговых артефактов (по умолчанию {DEFAULT_PROCESSED_DIR}).",
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
    universe_parser.add_argument(
        "--force",
        action="store_true",
        help="Явное подтверждение пересборки при существующем замке состава.",
    )
    universe_parser.set_defaults(handler=_cmd_universe)

    review_parser = subparsers.add_parser(
        "review",
        help="Ручная проверка состава: внести исключения по качеству данных.",
    )
    review_parser.add_argument(
        "--exclude",
        action="append",
        nargs=2,
        metavar=("SECID", "PRICHINA"),
        default=[],
        help="Исключить инструмент с указанной причиной (можно повторять).",
    )
    review_parser.set_defaults(handler=_cmd_review)

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
        help="Диагностика качества и корпоративные события.",
    )
    audit_parser.set_defaults(handler=_cmd_audit)

    collect_parser = subparsers.add_parser(
        "collect-events",
        help="Зафиксировать корпоративное событие (полуручной сбор, шаг A2).",
    )
    collect_parser.add_argument("--secid", required=True, help="Код инструмента.")
    collect_parser.add_argument("--event-date", required=True, help="Дата события (ISO).")
    collect_parser.add_argument(
        "--event-type",
        required=True,
        choices=["split"],
        help="Тип события.",
    )
    collect_parser.add_argument("--coef", type=float, help="Коэффициент приведения.")
    collect_parser.add_argument("--source", help="Происхождение сведения.")
    collect_parser.add_argument("--retrieved-at", help="Дата получения (ISO).")
    collect_parser.set_defaults(handler=_cmd_collect_events)

    panel_parser = subparsers.add_parser(
        "panel",
        help="Построение месячной панели и манифеста.",
    )
    panel_parser.set_defaults(handler=_cmd_panel)

    acceptance_parser = subparsers.add_parser(
        "acceptance",
        help="Приёмка этапа 8: детерминизм и отсутствие утечки.",
    )
    acceptance_parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("data/reports"),
        help="Каталог для отчёта приёмки (по умолчанию data/reports).",
    )
    acceptance_parser.set_defaults(handler=_cmd_acceptance)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Оценка модели и полезности факторов на построенной панели.",
    )
    evaluate_parser.set_defaults(handler=_cmd_evaluate)

    selection_parser = subparsers.add_parser(
        "selection",
        help="Отбор состава факторов и сравнение моделей A/B/C (модель C).",
    )
    selection_parser.set_defaults(handler=_cmd_selection)

    bootstrap_parser = subparsers.add_parser(
        "bootstrap",
        help="Парный блочный бутстрэп разности MSE моделей B и C.",
    )
    bootstrap_parser.set_defaults(handler=_cmd_bootstrap)

    sensitivity_parser = subparsers.add_parser(
        "sensitivity",
        help="Проверки устойчивости процедуры отбора (длина окна/блока, исключения).",
    )
    sensitivity_parser.set_defaults(handler=_cmd_sensitivity)

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