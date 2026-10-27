"""Общие проверки качества дневных данных.

Модуль слоя обрабатывающей логики: только локальные файлы, без сети.
Содержит набор проверок качества, каждая из которых возвращает перечень
замечаний в виде строк с идентификатором инструмента, датой, типом замечания
и значением.
"""

from __future__ import annotations

import itertools
from datetime import date

import pandas as pd

from equity_factor_study.calendars import build_ru_non_workdays

# Порог аномально большого дневного изменения (доля, например 0.5 = 50%).
DEFAULT_LARGE_CHANGE = 0.5
# Предельный календарный разрыв без торгов (дней).
DEFAULT_MAX_GAP_DAYS = 10
# Относительный допуск на расхождение агругированного объёма и суммы дневных значений.
DEFAULT_VOLUME_TOLERANCE = 0.01

# Типы замечаний.
NONPOSITIVE_PRICE = "nonpositive_price"
LARGE_DAY_CHANGE = "large_day_change"
TRADING_GAP = "trading_gap"
VOLUME_DISCREPANCY = "volume_discrepancy"
MISSING_PRICE_ON_TRADING_DAY = "missing_price_on_trading_day"

FINDING_COLUMNS = ["secid", "date", "type", "value"]


def _serialize_date(value: date | pd.Timestamp) -> str:
    return pd.Timestamp(value).date().isoformat()


def check_nonpositive_prices(table: pd.DataFrame) -> pd.DataFrame:
    """Выявить неположительные (нулевые и отрицательные) цены закрытия."""
    if table.empty or "close" not in table.columns:
        return pd.DataFrame(columns=FINDING_COLUMNS)
    mask = table["close"].notna() & (table["close"] <= 0)
    rows = table.loc[mask]
    return pd.DataFrame(
        {
            "secid": rows["secid"],
            "date": rows["trade_date"].map(_serialize_date),
            "type": NONPOSITIVE_PRICE,
            "value": rows["close"].astype(float),
        }
    ).reset_index(drop=True)


def check_large_changes(
    table: pd.DataFrame,
    *,
    threshold: float = DEFAULT_LARGE_CHANGE,
    event_dates: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Выявить аномально большие дневные изменения.

    Даты, совпадающие с задокументированным корпоративным событием (пара
    (secid, дата)), не помечаются как дефект — событие выдаётся отдельно, а
    ряд не исправляется порогом.
    """
    event_dates = event_dates or set()
    if table.empty or "close" not in table.columns:
        return pd.DataFrame(columns=FINDING_COLUMNS)

    table = table.sort_values(["secid", "trade_date"]).reset_index(drop=True)
    groups = [g.copy() for _, g in table.groupby("secid", sort=False)]

    findings: list[dict] = []
    for group in groups:
        prev_close = group["close"].shift(1)
        change = group["close"] / prev_close - 1.0
        for idx in group.index:
            change_value = change.loc[idx]
            if pd.isna(change_value):
                continue
            if abs(change_value) <= threshold:
                continue
            secid = group.loc[idx, "secid"]
            date_iso = _serialize_date(group.loc[idx, "trade_date"])
            if (secid, date_iso) in event_dates:
                continue
            findings.append(
                {
                    "secid": secid,
                    "date": date_iso,
                    "type": LARGE_DAY_CHANGE,
                    "value": float(change_value),
                }
            )

    if not findings:
        return pd.DataFrame(columns=FINDING_COLUMNS)
    return pd.DataFrame(findings).reset_index(drop=True)


def check_trading_gaps(
    table: pd.DataFrame,
    *,
    max_gap_days: int = DEFAULT_MAX_GAP_DAYS,
) -> pd.DataFrame:
    """Выявить длинные интервалы без торгов (календарный разрыв)."""
    if table.empty or "trade_date" not in table.columns:
        return pd.DataFrame(columns=FINDING_COLUMNS)

    findings: list[dict] = []
    for secid, group in table.groupby("secid", sort=True):
        dates = sorted(group["trade_date"].dropna().unique())
        for prev, current in itertools.pairwise(dates):
            gap = (current - prev).days
            if gap > max_gap_days:
                findings.append(
                    {
                        "secid": secid,
                        "date": _serialize_date(current),
                        "type": TRADING_GAP,
                        "value": float(gap),
                    }
                )

    if not findings:
        return pd.DataFrame(columns=FINDING_COLUMNS)
    return pd.DataFrame(findings).reset_index(drop=True)


def check_volume_discrepancy(
    table: pd.DataFrame,
    *,
    expected_volume: dict[str, float] | None = None,
    tolerance: float = DEFAULT_VOLUME_TOLERANCE,
) -> pd.DataFrame:
    """Сверить сумму дневных значений объёма с ожидаемым агрегатом.

    ``expected_volume`` — карта «инструмент -> ожидаемый агрегированный объём».
    Если карта не задана, проверка пропускается (нет эталона для сверки).
    """
    if not expected_volume:
        return pd.DataFrame(columns=FINDING_COLUMNS)
    if table.empty or "value" not in table.columns:
        return pd.DataFrame(columns=FINDING_COLUMNS)

    findings: list[dict] = []
    for secid, group in table.groupby("secid", sort=True):
        total = pd.to_numeric(group["value"], errors="coerce").sum()
        expected = expected_volume.get(secid)
        if expected is None:
            continue
        if expected == 0:
            deviation = 0.0 if total == 0 else float("inf")
        else:
            deviation = abs(total - expected) / expected
        if deviation > tolerance:
            findings.append(
                {
                    "secid": secid,
                    "date": _serialize_date(group["trade_date"].max()),
                    "type": VOLUME_DISCREPANCY,
                    "value": float(deviation),
                }
            )

    if not findings:
        return pd.DataFrame(columns=FINDING_COLUMNS)
    return pd.DataFrame(findings).reset_index(drop=True)


def check_missing_price_on_trading_day(
    table: pd.DataFrame,
    *,
    reference_calendar: set[date] | None = None,
) -> pd.DataFrame:
    """Выявить торговые дни, в которые отсутствует цена.

    Календарь для каждого инструмента ограничивается его собственным
    интервалом торгов (от первой до последней даты котировки): дни до дебюта
    инструмента не считаются пропусками. Из календаря исключаются нерабочие
    дни РФ (праздники, переносы), когда торги не идут.

    Если ``reference_calendar`` передан, он используется как база для всех
    инструментов (но всё равно ограничивается интервалом торгов каждого).
    """
    if table.empty or "trade_date" not in table.columns:
        return pd.DataFrame(columns=FINDING_COLUMNS)

    if reference_calendar is None:
        reference_calendar = set(table["trade_date"].dropna().unique())
    # Привести календарь к ``datetime.date``.
    reference_calendar = {d.date() if hasattr(d, "date") else d for d in reference_calendar}

    findings: list[dict] = []
    for secid, group in table.groupby("secid", sort=True):
        dates = group["trade_date"].dropna()
        if dates.empty:
            continue
        # Привести границы интервала торгов и даты к ``datetime.date``.
        first = dates.min()
        last = dates.max()
        first_date = first.date() if hasattr(first, "date") else first
        last_date = last.date() if hasattr(last, "date") else last
        present_with_price_dates: set[date] = set()
        for d, series in group.groupby("trade_date", sort=True):
            if series["close"].notna().any():
                present_with_price_dates.add(d.date() if hasattr(d, "date") else d)
        # Рабочие дни в интервале торгов инструмента (без праздников РФ).
        work_days = {
            d
            for d in reference_calendar
            if first_date <= d <= last_date and d not in build_ru_non_workdays(first_date, last_date)
        }
        for d in sorted(work_days):
            if d not in present_with_price_dates:
                findings.append(
                    {
                        "secid": secid,
                        "date": _serialize_date(d),
                        "type": MISSING_PRICE_ON_TRADING_DAY,
                        "value": 1.0,
                    }
                )

    if not findings:
        return pd.DataFrame(columns=FINDING_COLUMNS)
    return pd.DataFrame(findings).reset_index(drop=True)