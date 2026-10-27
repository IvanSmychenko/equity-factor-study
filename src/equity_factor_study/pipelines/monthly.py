"""Построение месячной панели (этап 7).

Модуль слоя обработки: работает только с локальными файлами, без сети.
Приводит нормализованные дневные данные к месячным наблюдениям по одной
строке на пару «инструмент — календарный месяц» и формирует итоговую панель и
манифест.

Схема панели и производные даты описаны в docs/protocol.md (пункт Б.6) и
docs/data_dictionary.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

PANEL_COLUMNS = [
    "secid",
    "month",
    "price_date",
    "month_end_price",
    "return_1m",
    "momentum_prev_3m",
    "volatility_3m",
    "key_rate_pct",
    "key_rate_change_pp",
    "target_return_next_1m",
    "quality_flags",
]


class MonthlyError(Exception):
    """Ошибка построения месячной панели."""


@dataclass
class PanelResult:
    panel: pd.DataFrame
    manifest: dict


def _month_start(value: date) -> date:
    return date(value.year, value.month, 1)


def _next_month(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def _shift_months(value: date, months: int) -> date:
    total = value.year * 12 + (value.month - 1) + months
    year, month_index = divmod(total, 12)
    return date(year, month_index + 1, 1)


def _last_valid_close(
    daily: pd.DataFrame,
    *,
    secid: str,
    month_start: date,
    max_trading_days: int,
    calendar: list[date],
) -> tuple[date | None, float | None, str]:
    """Последняя допустимая цена закрытия месяца с учётом допуска.

    Допустимая цена — цена на последний торговый день месяца (или ближайший к
    концу месяца), отстоящий от календарного конца месяца не более чем на
    ``max_trading_days`` торговых дней. При нарушении допуска цена пуста и
    выставляется пометка.
    """
    month_end = _shift_months(month_start, 1) - timedelta(days=1)

    # Торговые дни месяца.
    month_days = [d for d in calendar if month_start <= d <= month_end]
    if not month_days:
        return None, None, "no_trading_days_in_month"

    # Последний торговый день месяца, который есть в данных инструмента.
    sec_rows = daily[(daily["secid"] == secid) & (daily["trade_date"].isin(month_days))]
    if sec_rows.empty:
        return None, None, "no_trading_days_in_month"

    sec_rows = sec_rows.sort_values("trade_date")
    candidate_date = sec_rows["trade_date"].iloc[-1]
    candidate_close = sec_rows["close"].iloc[-1]

    if pd.isna(candidate_close):
        return None, None, "missing_close_price"

    # Допуск: число торговых дней от последнего торгового дня данных до конца месяца.
    # Посчитаем, сколько торговых дней календаря лежит строго после candidate_date.
    trailing_days = sum(1 for d in month_days if d > candidate_date)
    if trailing_days > max_trading_days:
        return candidate_date, None, "stale_price_too_far_from_month_end"

    return candidate_date, float(candidate_close), ""


def _monthly_price_series(
    daily: pd.DataFrame,
    *,
    secid: str,
    calendar: list[date],
    months: list[date],
    max_trading_days: int,
) -> pd.DataFrame:
    """Построить месячный ряд цен по инструменту."""
    rows: list[dict] = []
    for month in months:
        price_date, price, flag = _last_valid_close(
            daily,
            secid=secid,
            month_start=month,
            max_trading_days=max_trading_days,
            calendar=calendar,
        )
        rows.append(
            {
                "month": month,
                "price_date": price_date,
                "month_end_price": price,
                "flag": flag,
            }
        )
    return pd.DataFrame(rows)


def adjust_prices_for_splits(
    daily: pd.DataFrame,
    *,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Привести ряд цен к единому масштабу по корпоративным событиям (шаг A3).

    Для каждого задокументированного сплита/консолидации (``event_type='split'``
    с ``coef``) значения ``close`` до даты события умножаются на ``coef``:
    обратный сплит ВТБ 15.07.2024 (цена 0.02 → 92.95, ``coef``≈4663.8)
    приводит прежний масштаб к новому, устраняя фантомную доходность и
    неискажая коэффициент OLS. Ряд корректируется **официальным
    коэффициентом**, а не порогом; исходные данные не изменяются — правка
    применяется к копии, передаваемой в панель.

    Возвращает копию ``daily`` с приведёнными ценами.
    """
    out = daily.copy()
    if events is None or events.empty:
        return out
    if "secid" not in events.columns or "date" not in events.columns:
        raise MonthlyError("events-таблица должна содержать колонки secid, date")

    splits = events[events.get("event_type", pd.Series(dtype=str)) == "split"].copy()
    for _, row in splits.iterrows():
        coef = row.get("coef")
        if coef is None or pd.isna(coef) or coef <= 0:
            raise MonthlyError(
                f"Для сплита {row['secid']} {row['date']} не задан положительный coef"
            )
        secid = row["secid"]
        event_date = pd.Timestamp(row["date"]).date()
        mask = (out["secid"] == secid) & (out["trade_date"] < event_date)
        out.loc[mask, "close"] = out.loc[mask, "close"] * float(coef)
    return out


def build_monthly_panel(
    daily: pd.DataFrame,
    *,
    secids: list[str],
    from_month: date,
    to_month: date,
    max_trading_days: int,
    momentum_window: int,
    volatility_window: int,
    key_rate: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Построить месячную панель за период наблюдения.

    ``daily`` — нормализованная дневная таблица (secid, trade_date, close, value).
    ``key_rate`` — дневной ряд (date, rate). Возвращает панель с одной строкой
    на пару «инструмент — месяц».
    """
    if "secid" not in daily.columns or "trade_date" not in daily.columns or "close" not in daily.columns:
        raise MonthlyError("daily-таблица должна содержать колонки secid, trade_date, close")

    # Привести цены к единому масштабу по зафиксированным сплитам (шаг A3).
    daily = adjust_prices_for_splits(daily, events=events)

    calendar = sorted(set(daily["trade_date"].dropna()))
    months = [
        _shift_months(from_month, i)
        for i in range(_months_between(from_month, to_month) + 1)
    ]

    frames: list[pd.DataFrame] = []
    for secid in secids:
        per_sec = daily[daily["secid"] == secid].copy()
        if per_sec.empty:
            # Инструмент без данных в периоде — но в выборке.
            empty_rows = pd.DataFrame([_empty_row(secid, month) for month in months])
            empty_rows = _coerce_panel_dtypes(empty_rows)
            frames.append(empty_rows[PANEL_COLUMNS])
            continue

        monthly = _monthly_price_series(
            per_sec,
            secid=secid,
            calendar=calendar,
            months=months,
            max_trading_days=max_trading_days,
        )

        # Дневные доходности для волатильности.
        per_sec = per_sec.sort_values("trade_date")
        daily_ret = per_sec.set_index("trade_date")["close"].pct_change(fill_method=None)

        # Сдвиг целевой переменной: return_next_1m[month] = price[month+1]/price[month]-1.
        price_by_month = monthly[["month", "month_end_price"]].copy()
        price_by_month["next_price"] = price_by_month["month_end_price"].shift(-1)
        price_by_month["target_return"] = (
            price_by_month["next_price"] / price_by_month["month_end_price"] - 1.0
        )

        # Признаки по месячной цене.
        monthly["return_1m"] = monthly["month_end_price"].pct_change(fill_method=None)
        monthly["momentum_prev_3m"] = (
            monthly["month_end_price"].shift(1) / monthly["month_end_price"].shift(momentum_window + 1) - 1.0
        )
        monthly["volatility_3m"] = [
            _rolling_volatility(per_sec, m, daily_ret, volatility_window) for m in monthly["month"]
        ]

        monthly["target_return_next_1m"] = price_by_month["target_return"].values

        # Пометки качества.
        monthly["quality_flags"] = monthly["flag"].apply(_flags_to_string)
        monthly["quality_flags"] = monthly["quality_flags"].mask(
            monthly["return_1m"].isna() & (monthly["quality_flags"] == ""), "no_previous_price"
        )

        # Колонки ставки присутствуют всегда; заполняются при наличии ряда.
        monthly["key_rate_pct"] = None
        monthly["key_rate_change_pp"] = None
        if key_rate is not None and not key_rate.empty:
            monthly = _attach_key_rate(monthly, key_rate)

        # Оставляем только базовые месяцы, отбрасываем расширение для моментума.
        base = monthly[monthly["month"].isin(months)].copy()
        base["secid"] = secid
        base = base[PANEL_COLUMNS]
        frames.append(base)

    if not frames:
        return pd.DataFrame(columns=PANEL_COLUMNS)

    frames = [_coerce_panel_dtypes(f) for f in frames]
    panel = pd.concat(frames, ignore_index=True)
    panel = panel[PANEL_COLUMNS]
    panel = panel.sort_values(["secid", "month"]).reset_index(drop=True)
    return panel


def _coerce_panel_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Привести числовые колонки к float для согласованности dtypes при concat."""
    numeric_columns = [
        "month_end_price",
        "return_1m",
        "momentum_prev_3m",
        "volatility_3m",
        "key_rate_pct",
        "key_rate_change_pp",
        "target_return_next_1m",
    ]
    result = frame.copy()
    for col in numeric_columns:
        if col in result.columns:
            result[col] = result[col].astype(float)
    return result


def _months_between(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month)


def build_manifest(
    *,
    panel: pd.DataFrame,
    protocol_version: str,
    observation_start: date,
    observation_end: date,
    secids: list[str],
    source_files: list[dict],
    code_version: str,
    dependencies: dict,
    adjustments: list[dict] | None = None,
) -> dict:
    """Собрать манифест происхождения панели.

    ``adjustments`` — перечень применённых к ряду корректировок (например,
    приведение цен к единому масштабу по сплиту/консолидации). Каждая запись:
    secid, date, event_type, coef, source.
    """
    manifest = {
        "protocol_version": protocol_version,
        "observation_start": observation_start.isoformat(),
        "observation_end": observation_end.isoformat(),
        "instruments": secids,
        "source_files": source_files,
        "code_version": code_version,
        "dependencies": dependencies,
        "rows": len(panel),
    }
    if adjustments:
        manifest["adjustments"] = adjustments
    return manifest


def _empty_row(secid: str, month: date) -> dict:
    return {
        "secid": secid,
        "month": month,
        "price_date": None,
        "month_end_price": None,
        "return_1m": None,
        "momentum_prev_3m": None,
        "volatility_3m": None,
        "key_rate_pct": None,
        "key_rate_change_pp": None,
        "target_return_next_1m": None,
        "quality_flags": "no_trading_days_in_month",
    }


def _rolling_volatility(
    per_sec: pd.DataFrame,
    month: date,
    daily_ret: pd.Series,
    window: int,
) -> float | None:
    """Стандартное отклонение дневных доходностей за окно календарных месяцев."""
    month_end = _shift_months(month, 1) - timedelta(days=1)
    month_start = _shift_months(month, -(window - 1))
    mask = (per_sec["trade_date"] >= month_start) & (per_sec["trade_date"] <= month_end)
    returns = daily_ret.loc[per_sec.loc[mask, "trade_date"]]
    returns = returns.dropna()
    if returns.empty:
        return None
    return float(returns.std(ddof=1))


def _flags_to_string(flag: str) -> str:
    return flag if flag else ""


def _attach_key_rate(monthly: pd.DataFrame, key_rate: pd.DataFrame) -> pd.DataFrame:
    """Присоединить действующую ставку и её изменение к месячным строкам."""
    kr = key_rate.copy()
    kr["date"] = pd.to_datetime(kr["date"])
    kr["month"] = kr["date"].dt.to_period("M").dt.to_timestamp()
    # Последнее значение действующей ставки внутри месяца.
    kr_last = kr.sort_values("date").groupby("month")["rate"].last().reset_index()
    kr_last = kr_last.rename(columns={"rate": "key_rate_pct"})
    kr_last = kr_last.sort_values("month").reset_index(drop=True)
    kr_last["key_rate_change_pp"] = kr_last["key_rate_pct"] - kr_last["key_rate_pct"].shift(1)

    mapping = kr_last.set_index("month")
    monthly["key_rate_pct"] = monthly["month"].map(mapping["key_rate_pct"])
    monthly["key_rate_change_pp"] = monthly["month"].map(mapping["key_rate_change_pp"])
    return monthly