"""Загрузка ряда ключевой ставки (этап 5).

Модуль слоя сбора: разрешены сетевые вызовы. Выполняет полную загрузку
дневного ряда ключевой ставки за период наблюдения и нормализует его в
дневную таблицу с едиными типами. Результат сохраняется в ``data/interim``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from equity_factor_study.calendars import build_ru_non_workdays
from equity_factor_study.clients.cbr import CBRClient, KeyRateDownload

# Целевые колонки нормализованного ряда ставки.
DAILY_COLUMNS = ["date", "rate"]


class KeyRateError(Exception):
    """Ошибка загрузки и нормализации ряда ключевой ставки."""


@dataclass
class KeyRateSummary:
    rows: int
    date_from: date | None
    date_to: date | None
    gaps: int
    complete: bool


@dataclass
class KeyRateSeries:
    table: pd.DataFrame
    summary: KeyRateSummary
    raw_paths: list[Path]

    @property
    def rows(self) -> int:
        return len(self.table)


def normalize_key_rate(table: pd.DataFrame) -> pd.DataFrame:
    """Привести ряд ставки к единым типам: ``date`` (date) и ``rate`` (float)."""
    if table.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)

    out = table.copy()
    out["date"] = out["date"].map(lambda d: d if isinstance(d, date) else pd.to_datetime(d).date())
    out["rate"] = pd.to_numeric(out["rate"], errors="coerce")
    out = out.drop_duplicates(subset=["date"], keep="first")
    out = out.sort_values("date").reset_index(drop=True)
    return out[DAILY_COLUMNS]


def _business_days(from_date: date, to_date: date) -> set[date]:
    """Множество рабочих дней (будни без праздников РФ) в интервале.

    Используется как ориентир покрытия ряда ставки: нерабочие дни
    (праздники, переносы, особые дни) не считаются разрывом, поскольку
    ставка в них не публикуется.
    """
    result: set[date] = set()
    current = from_date
    while current <= to_date:
        if current.weekday() < 5:
            result.add(current)
        current = current + timedelta(days=1)
    return result - build_ru_non_workdays(from_date, to_date)


def summarize_key_rate(table: pd.DataFrame, from_date: date, to_date: date) -> KeyRateSummary:
    """Сформировать сводку покрытия ряда ставки за период."""
    if table.empty:
        return KeyRateSummary(rows=0, date_from=None, date_to=None, gaps=0, complete=False)

    dates = sorted(set(table["date"].dropna()))
    observed_set = set(dates)

    # Пропущенные будние дни внутри интервала наблюдения считаются разрывом.
    baseline = _business_days(from_date, to_date)
    gaps = sum(1 for d in baseline if d not in observed_set)

    complete = gaps == 0 and len(dates) > 0
    return KeyRateSummary(
        rows=len(dates),
        date_from=dates[0],
        date_to=dates[-1],
        gaps=gaps,
        complete=complete,
    )


def collect_key_rate(
    client: CBRClient,
    *,
    from_date: date,
    to_date: date,
) -> KeyRateSeries:
    """Загрузить ряд ключевой ставки за период и нормализовать его."""
    download: KeyRateDownload = client.get_key_rate(from_date=from_date, to_date=to_date)
    table = normalize_key_rate(download.table)
    return KeyRateSeries(
        table=table,
        summary=summarize_key_rate(table, from_date, to_date),
        raw_paths=[download.raw_path],
    )


def write_daily_key_rate(table: pd.DataFrame, path: Path) -> Path:
    """Записать нормализованный ряд ставки в CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = table.copy()
    if not out.empty:
        out["date"] = out["date"].map(lambda d: d.isoformat() if isinstance(d, date) else d)
    out.to_csv(path, index=False)
    return path