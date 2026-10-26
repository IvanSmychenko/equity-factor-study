"""Загрузка дневных цен по выборке (этап 5).

Модуль слоя сбора: разрешены сетевые вызовы. Выполняет полную загрузку
дневной истории по отобранным инструментам за весь период наблюдения,
отбрасывает строки, не относящиеся к выбранному режиму торгов, и приводит
данные к единым типам и единицам измерения. Результат нормализуется в
дневную таблицу и сохраняется в ``data/interim``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from equity_factor_study.clients.moex import HistoryDownload, MOEXClient

# Целевые колонки нормализованной дневной таблицы.
DAILY_COLUMNS = ["secid", "trade_date", "close", "value"]


class PricesError(Exception):
    """Ошибка загрузки и нормализации дневных цен."""


@dataclass
class InstrumentSummary:
    secid: str
    rows: int
    date_from: date | None
    date_to: date | None
    missing_trading_days: int
    complete: bool


@dataclass
class PricesDownload:
    table: pd.DataFrame
    summaries: list[InstrumentSummary]
    raw_paths: list[Path]

    @property
    def rows(self) -> int:
        return len(self.table)


def _to_number(series: pd.Series) -> pd.Series:
    """Привести числовой столбец к float, обработав разделитель-запятую."""
    if series.empty:
        return series.astype(float)

    def _convert(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return value
        if isinstance(value, str):
            stripped = value.strip().replace("\u00a0", "").replace(" ", "")
            if "," in stripped and "." not in stripped:
                stripped = stripped.replace(",", ".")
            value = stripped
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    return series.map(_convert).astype(float)


def _to_date(value) -> date | None:
    """Привести значение даты к ``datetime.date``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return pd.to_datetime(value).date()


def normalize_history(table: pd.DataFrame, *, board: str) -> pd.DataFrame:
    """Нормализовать дневную историю MOEX в единую таблицу.

    Оставляет только строки выбранного режима торгов, приводит даты к единому
    формату и числа к числовому типу, удаляет дубли по паре
    «инструмент — дата». Возвращает таблицу с колонками ``secid``,
    ``trade_date``, ``close``, ``value``.
    """
    if table.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)

    required = {"SECID", "TRADEDATE"}
    missing = required - set(table.columns)
    if missing:
        raise PricesError(f"В истории отсутствуют обязательные колонки: {sorted(missing)}")

    if "BOARDID" in table.columns:
        table = table[table["BOARDID"] == board]

    rows: list[dict] = []
    for record in table.to_dict("records"):
        secid = record.get("SECID")
        trade_date = _to_date(record.get("TRADEDATE"))
        close = _to_number(pd.Series([record.get("CLOSE")])).iloc[0] if "CLOSE" in record else None
        value = _to_number(pd.Series([record.get("VALUE")])).iloc[0] if "VALUE" in record else None
        if secid is None or trade_date is None:
            continue
        rows.append(
            {
                "secid": str(secid),
                "trade_date": trade_date,
                "close": close,
                "value": value,
            }
        )

    if not rows:
        return pd.DataFrame(columns=DAILY_COLUMNS)

    normalized = pd.DataFrame(rows, columns=DAILY_COLUMNS)
    normalized = normalized.drop_duplicates(subset=["secid", "trade_date"], keep="first")
    normalized = normalized.sort_values(["secid", "trade_date"]).reset_index(drop=True)
    return normalized


def _reference_calendar(table: pd.DataFrame) -> set[date]:
    """Календарь торговых дней из объединения всех наблюдаемых дат."""
    if table.empty:
        return set()
    return set(table["trade_date"].dropna())


def summarize(table: pd.DataFrame) -> list[InstrumentSummary]:
    """Сформировать сводку загрузки по каждому инструменту.

    Статус завершённости определяется отсутствием пропусков торговых дней
    относительно общего календаря выборки.
    """
    if table.empty:
        return []

    calendar = _reference_calendar(table)
    summaries: list[InstrumentSummary] = []
    for secid, group in table.groupby("secid"):
        dates = set(group["trade_date"].dropna())
        missing = len(calendar - dates)
        summaries.append(
            InstrumentSummary(
                secid=str(secid),
                rows=len(group),
                date_from=min(dates) if dates else None,
                date_to=max(dates) if dates else None,
                missing_trading_days=missing,
                complete=missing == 0,
            )
        )
    summaries.sort(key=lambda item: item.secid)
    return summaries


def collect_prices(
    client: MOEXClient,
    *,
    secids: list[str],
    board: str,
    from_date: date,
    to_date: date,
) -> PricesDownload:
    """Загрузить полную дневную историю по каждому инструменту и нормализовать."""
    tables: list[pd.DataFrame] = []
    raw_paths: list[Path] = []
    for secid in secids:
        download: HistoryDownload = client.get_full_history(
            security=secid,
            from_date=from_date,
            till_date=to_date,
            board=board,
        )
        raw_paths.extend(download.raw_paths)
        normalized = normalize_history(download.table, board=board)
        tables.append(normalized)

    if not tables:
        table = pd.DataFrame(columns=DAILY_COLUMNS)
    else:
        table = pd.concat(tables, ignore_index=True)
        if not table.empty:
            table = table.sort_values(["secid", "trade_date"]).reset_index(drop=True)

    return PricesDownload(table=table, summaries=summarize(table), raw_paths=raw_paths)


def write_daily_prices(table: pd.DataFrame, path: Path) -> Path:
    """Записать нормализованную дневную таблицу цен в CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = table.copy()
    if not out.empty:
        out["trade_date"] = out["trade_date"].map(lambda d: d.isoformat() if isinstance(d, date) else d)
    out.to_csv(path, index=False)
    return path