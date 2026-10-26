"""Диагностика качества и корпоративные события (этап 6).

Модуль слоя обработки: работает только с локальными файлами, не обращается к
сети. Применяет набор проверок качества к нормализованной дневной таблице цен
и формирует перечень замечаний, сводку качества и перечень корпоративных
событий.

Корпоративные события выявляются по данным о корпоративных действиях
(отдельная таблица-артефакт), а не по величине скачка цены. Ряд не
исправляется единым порогом изменения цены: событие помечается отдельно,
а доходности считаются по исходным ценам.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from equity_factor_study.validation import (
    FINDING_COLUMNS,
    check_large_changes,
    check_missing_price_on_trading_day,
    check_nonpositive_prices,
    check_trading_gaps,
    check_volume_discrepancy,
)

# Колонки таблицы корпоративных событий (артефакт-вход).
EVENT_COLUMNS = ["secid", "date", "event_type"]

# Типы корпоративных событий.
EVENT_SPLIT = "split"
EVENT_DIVIDEND = "dividend"
EVENT_RIGHTS = "rights_issue"


class AuditError(Exception):
    """Ошибка диагностики качества."""


@dataclass
class AuditResult:
    findings: pd.DataFrame
    events: pd.DataFrame
    summary: pd.DataFrame


def _as_date(value) -> date:
    return pd.Timestamp(value).date()


def load_corporate_events(path: Path) -> pd.DataFrame:
    """Загрузить таблицу корпоративных событий из CSV."""
    if not path.exists():
        return pd.DataFrame(columns=EVENT_COLUMNS)
    events = pd.read_csv(path)
    for column in EVENT_COLUMNS:
        if column not in events.columns:
            raise AuditError(f"В таблице событий нет колонки {column!r}")
    events["date"] = events["date"].map(_as_date)
    return events[EVENT_COLUMNS]


def build_events_index(events: pd.DataFrame) -> set[tuple[str, str]]:
    """Построить множество пар (secid, дата) задокументированных событий."""
    if events.empty:
        return set()
    return {
        (str(row["secid"]), pd.Timestamp(row["date"]).date().isoformat())
        for row in events.to_dict("records")
    }


def run_audit(
    prices: pd.DataFrame,
    *,
    events: pd.DataFrame | None = None,
    expected_volume: dict[str, float] | None = None,
    large_change_threshold: float = 0.5,
    max_gap_days: int = 10,
    volume_tolerance: float = 0.01,
) -> AuditResult:
    """Прогнать все проверки качества и собрать сводку и события."""
    events = events if events is not None else pd.DataFrame(columns=EVENT_COLUMNS)
    event_index = build_events_index(events)

    findings_parts = [
        check_nonpositive_prices(prices),
        check_large_changes(
            prices,
            threshold=large_change_threshold,
            event_dates=event_index,
        ),
        check_trading_gaps(prices, max_gap_days=max_gap_days),
        check_volume_discrepancy(
            prices,
            expected_volume=expected_volume,
            tolerance=volume_tolerance,
        ),
        check_missing_price_on_trading_day(prices),
    ]
    findings_parts = [part for part in findings_parts if not part.empty]
    if findings_parts:
        findings = pd.concat(findings_parts, ignore_index=True)
    else:
        findings = pd.DataFrame(columns=FINDING_COLUMNS)

    summary = _build_summary(findings)
    return AuditResult(findings=findings, events=events, summary=summary)


def _build_summary(findings: pd.DataFrame) -> pd.DataFrame:
    """Сводка качества: счётчики замечаний по типам."""
    if findings.empty:
        return pd.DataFrame(columns=["type", "count"])
    counts = findings["type"].value_counts().reset_index()
    counts.columns = ["type", "count"]
    return counts.reset_index(drop=True)


def write_findings(findings: pd.DataFrame, path: Path) -> Path:
    """Записать перечень замечаний."""
    path.parent.mkdir(parents=True, exist_ok=True)
    findings.to_csv(path, index=False)
    return path


def write_events(events: pd.DataFrame, path: Path) -> Path:
    """Записать перечень корпоративных событий."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not events.empty:
        out = events.copy()
        out["date"] = out["date"].map(lambda d: pd.Timestamp(d).date().isoformat())
    else:
        out = events
    out.to_csv(path, index=False)
    return path


def write_summary(summary: pd.DataFrame, path: Path) -> Path:
    """Записать сводку качества."""
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return path