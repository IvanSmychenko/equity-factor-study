"""Заведомые дефекты для проверки самих проверок (А.6).

Каждый дефект — это вариант построителя панели с одной внесённой ошибкой
сдвига/окна. Дефектный построитель сначала строит корректную панель через
`build_monthly_panel`, а затем искажает один из признаков так, как если бы в
исходный код была внесена конкретная ошибка. Тесты проверяют, что дефектный
результат отличается от эталонного по тому признаку, который ловится
соответствующим правилом (то есть что проверки утечки способны обнаружить
внесённый дефект).
"""

from __future__ import annotations

from equity_factor_study.pipelines.monthly import build_monthly_panel


def _panel(**kwargs):
    return build_monthly_panel(**kwargs)


def defect_return_uses_next_month(daily, *, secids, from_month, to_month,
                                  max_trading_days, momentum_window, volatility_window,
                                  key_rate=None):
    """Дефект D1: return_1m считается по цене следующего месяца (t+1/t − 1)."""
    frame = _panel(
        daily=daily, secids=secids, from_month=from_month, to_month=to_month,
        max_trading_days=max_trading_days, momentum_window=momentum_window,
        volatility_window=volatility_window, key_rate=key_rate,
    )
    frame = frame.sort_values(["secid", "month"])
    # ОШИБКА: сдвиг на −1 (следующий месяц) вместо предыдущего.
    frame["return_1m"] = frame.groupby("secid")["month_end_price"].transform(
        lambda s: s.shift(-1) / s - 1.0
    )
    return frame


def defect_momentum_window_3(daily, *, secids, from_month, to_month,
                             max_trading_days, momentum_window, volatility_window,
                             key_rate=None):
    """Дефект D2: моментум по окну t-3 вместо t-4 (сдвиг momentum_window)."""
    frame = _panel(
        daily=daily, secids=secids, from_month=from_month, to_month=to_month,
        max_trading_days=max_trading_days, momentum_window=momentum_window,
        volatility_window=volatility_window, key_rate=key_rate,
    )
    frame = frame.sort_values(["secid", "month"])
    frame["momentum_prev_3m"] = frame.groupby("secid")["month_end_price"].transform(
        lambda s: s.shift(1) / s.shift(momentum_window) - 1.0
    )
    return frame


def defect_volatility_full_period(daily, *, secids, from_month, to_month,
                                  max_trading_days, momentum_window, volatility_window,
                                  key_rate=None):
    """Дефект D3: волатильность по всему периоду вместо трёх месяцев."""
    frame = _panel(
        daily=daily, secids=secids, from_month=from_month, to_month=to_month,
        max_trading_days=max_trading_days, momentum_window=momentum_window,
        volatility_window=volatility_window, key_rate=key_rate,
    )
    # Перенести волатильность последнего месяца на все предыдущие (модель
    # «всё посчитано по всему периоду»).
    frame = frame.sort_values(["secid", "month"])
    frame["volatility_3m"] = frame.groupby("secid")["volatility_3m"].transform(
        lambda s: s.ffill().bfill()
    )
    return frame


def defect_key_rate_next_month(daily, *, secids, from_month, to_month,
                               max_trading_days, momentum_window, volatility_window,
                               key_rate=None):
    """Дефект D4: ставка берётся на конец следующего месяца."""
    frame = _panel(
        daily=daily, secids=secids, from_month=from_month, to_month=to_month,
        max_trading_days=max_trading_days, momentum_window=momentum_window,
        volatility_window=volatility_window, key_rate=key_rate,
    )
    frame = frame.sort_values(["secid", "month"])
    frame["key_rate_pct"] = frame.groupby("secid")["key_rate_pct"].shift(-1)
    return frame


DEFECTS = [
    ("D-RETURN-NEXT", defect_return_uses_next_month, "return_1m"),
    ("D-MOMENTUM-3", defect_momentum_window_3, "momentum_prev_3m"),
    ("D-VOLATILITY-FULL", defect_volatility_full_period, "volatility_3m"),
    ("D-KEYRATE-NEXT", defect_key_rate_next_month, "key_rate_pct"),
]

# Признак, контролируемый каждым дефектом, и функция проверки (в tests), которая
# обязана его обнаружить. Используется тестами А.6 для выбора канала внедрения.
DETECTORS = {
    "D-RETURN-NEXT": "a4",
    "D-MOMENTUM-3": "tl2",
    "D-VOLATILITY-FULL": "tl2",
    "D-KEYRATE-NEXT": "tl4",
}
