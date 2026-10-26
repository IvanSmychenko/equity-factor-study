"""Синтетические данные для проверок этапа 8 (детерминизм, утечка).

Набор по требованиям А.2: три инструмента, двадцать четыре месяца, цены
заданы вручную и намеренно немонотонны, чтобы перепутанное направление сдвига
проявлялось в знаке расхождения. Используется тестами и командой acceptance.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

MONTHS_IN_SAMPLE = 24
START_MONTH = date(2014, 1, 1)
SECIDS = ["SBER", "GAZP", "LKOH"]


def _shift_months(value: date, months: int) -> date:
    total = value.year * 12 + (value.month - 1) + months
    year, month_index = divmod(total, 12)
    return date(year, month_index + 1, 1)


def _month_iter(count: int, start: date = START_MONTH):
    for i in range(count):
        yield _shift_months(start, i)


def _last_trading_day(month_start: date) -> date:
    """Последний торговый день месяца — среда, отстоящая ≤5 дней от конца."""
    if month_start.month == 12:
        next_month = date(month_start.year + 1, 1, 1)
    else:
        next_month = date(month_start.year, month_start.month + 1, 1)
    month_end = next_month - timedelta(days=1)
    return month_end - timedelta(days=1)  # условный «предпоследний» торговый день


def build_synthetic_daily() -> pd.DataFrame:
    """Дневные цены трёх инструментов за 24 месяца, немонотонные.

    Цена каждого месяца задаётся явно как смещение от базовой и меняет знак
    прироста, чтобы перепутанный сдвиг доходности давал расхождение по знаку.
    """
    rows: list[dict] = []
    base = 100.0
    for idx, secid in enumerate(SECIDS):
        price = base * (1.0 + 0.1 * idx)
        for month in _month_iter(MONTHS_IN_SAMPLE):
            # Немонотонный ход: чередуем рост и падение.
            step = 1.0 if (month.month % 2 == 1) else -1.0
            factor = 1.0 + 0.02 * step + 0.003 * (month.month % 5)
            price = price * factor
            trade_date = _last_trading_day(month)
            rows.append(
                {
                    "secid": secid,
                    "trade_date": trade_date,
                    "close": round(price, 4),
                    "value": 1000.0 * (idx + 1),
                }
            )
            # Дополнительный торговый день в начале месяца для волатильности.
            mid_date = trade_date - timedelta(days=3)
            rows.append(
                {
                    "secid": secid,
                    "trade_date": mid_date,
                    "close": round(price * (1.0 + 0.001 * step), 4),
                    "value": 1000.0 * (idx + 1),
                }
            )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["secid", "trade_date"]).reset_index(drop=True)


def build_dense_weekday_daily(secid: str, months: int = 6, start: date | None = None) -> pd.DataFrame:
    """Плотный ряд: каждый будний день месяцу (для проверок допуска по концу месяца).

    ``months`` — число месяцев, ``start`` — первый месяц. Возвращает таблицу с
    будними днями как торговыми днями.
    """
    if start is None:
        start = date(2014, 1, 1)
    rows = []
    for month in _month_iter(months, start):
        next_month = _shift_months(month, 1)
        month_end = next_month - timedelta(days=1)
        cursor = month
        while cursor <= month_end:
            if cursor.weekday() < 5:  # будний день.
                rows.append({"secid": secid, "trade_date": cursor, "close": 100.0, "value": 1.0})
            cursor += timedelta(days=1)
    return pd.DataFrame(rows)


def build_synthetic_key_rate() -> pd.DataFrame:
    """Ключевая ставка: одно изменение в месяц (фиксировано в последний день)."""
    rows = []
    for month in _month_iter(MONTHS_IN_SAMPLE):
        rate = 5.5 + 0.25 * (month.month % 6)
        trade_date = _last_trading_day(month)
        rows.append({"date": trade_date, "rate": rate})
    return pd.DataFrame(rows)


def build_synthetic_panel(**overrides):
    """Построить панель на синтетическом наборе (удобный ярлык для тестов)."""
    from equity_factor_study.pipelines.monthly import build_monthly_panel

    final_month = list(_month_iter(MONTHS_IN_SAMPLE))[-1]
    return build_monthly_panel(
        build_synthetic_daily(),
        secids=SECIDS,
        from_month=START_MONTH,
        to_month=final_month,
        max_trading_days=5,
        momentum_window=3,
        volatility_window=3,
        key_rate=build_synthetic_key_rate(),
        **overrides,
    )


def build_long_continuous_panel(
    *,
    start: date = date(2014, 1, 1),
    end: date = date(2026, 9, 1),
    secids: list[str] | None = None,
    seed: int = 7,
) -> pd.DataFrame:
    """Непрерывная панель на весь период наблюдения (для модельного слоя).

    Генерирует детерминированный ряд цены для нескольких инструментов на
    каждый месяц периода с полным покрытием (без пропусков). Используется,
    чтобы упражнять обучающее окно 48 месяцев и контрольные даты Б.6.
    """
    import random

    from equity_factor_study.pipelines.monthly import build_monthly_panel

    secids = secids or ["A", "B"]
    rng = random.Random(seed)
    daily_rows = []
    for idx, secid in enumerate(secids):
        price = 100.0 * (1.0 + 0.05 * idx)
        for month in _month_iter(_months_between(start, end) + 1, start):
            step = 1.0 if (month.month % 2 == 1) else -1.0
            factor = 1.0 + 0.015 * step + 0.002 * rng.random()
            price = price * factor
            trade_date = _last_trading_day(month)
            daily_rows.append(
                {"secid": secid, "trade_date": trade_date, "close": round(price, 4), "value": 1.0}
            )
    daily = pd.DataFrame(daily_rows)

    # Детерминированная ключевая ставка на весь период.
    kr_rows = []
    for month in _month_iter(_months_between(start, end) + 1, start):
        kr_rows.append({"date": _last_trading_day(month), "rate": 5.5 + 0.5 * (month.month % 5)})
    key_rate = pd.DataFrame(kr_rows)

    return build_monthly_panel(
        daily,
        secids=secids,
        from_month=start,
        to_month=end,
        max_trading_days=5,
        momentum_window=3,
        volatility_window=3,
        key_rate=key_rate,
    )


def _months_between(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month)