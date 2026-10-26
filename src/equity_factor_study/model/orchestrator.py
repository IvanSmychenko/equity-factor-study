"""Модельный слой: оркестрация обучения и оценка полезности факторов.

Собирает пайплайн: панель → матрица наблюдений (единая маска) → скользящее
обучение → метрики по срезам → сравнение моделей с полным и сокращённым
набором факторов. Модуль не обращается к сети и не изменяет исходные данные.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from .evaluate import MetricResult, evaluate_forecasts, factor_usefulness
from .fit import walk_forward
from .panel import ObservationMatrix, build_observation_matrix


@dataclass(frozen=True)
class ModelReport:
    """Результат оценки модельного слоя."""

    feature_columns: list[str]
    full_metrics: MetricResult
    factor_usefulness: dict


def run_model_evaluation(
    panel: pd.DataFrame,
    *,
    feature_columns: list[str],
    training_window_months: int,
    refit_frequency: str,
    first_target_month: date,
    development_target_end: date,
    last_month: date,
) -> ModelReport:
    """Обучить полный набор факторов и вычислить метрики по срезам.

    Возвращает отчёт с метрикой полного набора на срезе оценки и сравнением
    с моделью без каждого фактора. Строки наблюдаений (единая маска) при
    сокращении набора остаются прежними (инвариант 5).
    """
    matrix = build_observation_matrix(panel)
    full_frame = _to_frame(matrix, feature_columns)

    full_forecasts = walk_forward(
        full_frame,
        feature_columns=feature_columns,
        training_window_months=training_window_months,
        refit_frequency=refit_frequency,
        first_target_month=first_target_month,
        last_month=last_month,
    )
    full_eval = evaluate_forecasts(
        full_forecasts,
        segment="evaluation",
        development_target_end=development_target_end,
    )

    usefulness = {}
    for factor in feature_columns:
        reduced_cols = [c for c in feature_columns if c != factor]
        reduced_forecasts = walk_forward(
            full_frame,
            feature_columns=reduced_cols,
            training_window_months=training_window_months,
            refit_frequency=refit_frequency,
            first_target_month=first_target_month,
            last_month=last_month,
        )
        reduced_eval = evaluate_forecasts(
            reduced_forecasts,
            segment="evaluation",
            development_target_end=development_target_end,
        )
        usefulness[factor] = factor_usefulness(full_eval, reduced_eval)

    return ModelReport(
        feature_columns=feature_columns,
        full_metrics=full_eval,
        factor_usefulness=usefulness,
    )


def _to_frame(matrix: ObservationMatrix, feature_columns: list[str]) -> pd.DataFrame:
    frame = matrix.features[feature_columns].copy()
    frame["target_return_next_1m"] = matrix.target.to_numpy()
    frame["month"] = matrix.identity["month"].to_numpy()
    frame["secid"] = matrix.identity["secid"].to_numpy()
    return frame


def load_panel(path: Path) -> pd.DataFrame:
    """Загрузить месячную панель из CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"панель не найдена: {path.resolve()}")
    frame = pd.read_csv(path)
    frame["month"] = pd.to_datetime(frame["month"])
    return frame