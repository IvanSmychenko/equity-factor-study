"""Модельный слой: агрегация метрик точности и оценка полезности факторов.

Агрегация по протоколу `evaluation.aggregation = "securities_then_months"`:
сначала ошибка усредняется по инструментам внутри месяца, затем по месяцам.
Первичная метрика `mse`, вторичная `mae`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .fit import Forecast


@dataclass(frozen=True)
class MetricResult:
    """Метрика точности на одном срезе (разработка или оценка)."""

    segment: str
    forecast_count: int
    mse: float
    mae: float
    aggregation: str = "securities_then_months"


def _flatten(forecasts: list[Forecast]) -> pd.DataFrame:
    rows = []
    for forecast in forecasts:
        n = len(forecast.predicted)
        actual = _as_list(forecast.actual, n)
        pred = _as_list(forecast.predicted, n)
        rows.append(
            {
                "month": forecast.forecast_month,
                "actual": actual,
                "predicted": pred,
            }
        )
    return pd.DataFrame(rows)


def _as_list(values, n):
    """Привести предсказание/факт к списку длины n."""
    if isinstance(values, pd.Series):
        arr = values.to_numpy(dtype=float)
    elif isinstance(values, pd.DataFrame):
        arr = values.iloc[:, 0].to_numpy(dtype=float)
    else:
        arr = np.asarray(values, dtype=float)
    return list(arr.reshape(-1))


def _metrics_from_lists(actuals: list[float], preds: list[float]) -> tuple[float, float]:
    actual = np.asarray(actuals, dtype=float)
    pred = np.asarray(preds, dtype=float)
    if len(actual) == 0:
        return float("nan"), float("nan")
    err = actual - pred
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    return mse, mae


def evaluate_forecasts(
    forecasts: list[Forecast],
    *,
    segment: str,
    development_target_end: date,
) -> MetricResult:
    """Вычислить агрегированные метрики `securities_then_months`.

    Сначала ошибка усредняется по инструментам (внутри месяца), затем по
    месяцам. Классификация месяца на «разработка»/«оценка» выполняется по
    ``development_target_end``: месяцы <= конца разработки — разработка,
    иначе — оценка.
    """
    frame = _flatten(forecasts)
    if frame.empty:
        return MetricResult(segment=segment, forecast_count=0, mse=float("nan"),
                            mae=float("nan"))

    # Средняя ошибка по инструментам внутри месяца.
    frame["month"] = pd.to_datetime(frame["month"])
    monthly = frame.groupby("month").apply(
        lambda group: _metrics_from_lists(
            group["actual"].iloc[0],
            group["predicted"].iloc[0],
        ),
        include_groups=False,
    )
    # Если месяц один, значение остаётся кортежем; иначе развернуть.
    if len(monthly) == 1:
        mses = [monthly.iloc[0][0]]
        maes = [monthly.iloc[0][1]]
    else:
        mses = [m[0] for m in monthly]
        maes = [m[1] for m in monthly]

    mse = float(np.mean(mses)) if mses else float("nan")
    mae = float(np.mean(maes)) if maes else float("nan")
    return MetricResult(
        segment=segment,
        forecast_count=len(frame),
        mse=mse,
        mae=mae,
    )


def factor_usefulness(
    full_metrics: MetricResult,
    reduced_metrics: MetricResult,
    *,
    primary_metric: str = "mse",
) -> dict:
    """Сравнить модель с полным набором признаков и с исключённым фактором.

    Полезность фактора — прирост ошибки (в %) при его удалении на срезе
    оценки. Положительный прирост (reduced хуже полного) означает, что фактор
    полезен вне выборки; отрицательный — что фактор не улучшает или вредит.
    """
    full_value = getattr(full_metrics, primary_metric)
    reduced_value = getattr(reduced_metrics, primary_metric)
    if full_value is None or reduced_value is None or np.isnan(full_value):
        return {"usefulness_pct": float("nan"), "direction": "indeterminate"}
    delta = (reduced_value - full_value) / full_value * 100.0
    return {
        "usefulness_pct": round(float(delta), 4),
        "direction": "useful" if delta > 0 else "harmful" if delta < 0 else "neutral",
    }