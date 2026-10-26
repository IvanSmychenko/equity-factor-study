"""Модельный слой: нормировка на обучении и скользящее регрессионное обучение.

Модуль реализует линейную модель (OLS) и скользящее переобучение по правилу
пункта Б.6: прогноз на целевой месяц t+1 использует оконные пары
«признаки месяца s — доходность месяца s+1» для ppа s из [t−W, t−1], где W —
длина окна обучения. Первый прогноз возможен в месяце s₀ + W + 1.

Инвариант 6: масштабирование признаков вычисляется только в пределах
обучающего окна и применяется к наблюдениям вне окна без пересчёта.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .panel import TARGET_COLUMN


class FitError(Exception):
    """Ошибка обучения модели."""


@dataclass(frozen=True)
class StandardScaler:
    """Масштабирование признаков (z-нормировка) на обучающих данных.

    Параметры фиксируются во время обучения и не пересчитываются при
    применении (инвариант 6).
    """

    mean: pd.Series
    scale: pd.Series

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return (features - self.mean) / self.scale

    @staticmethod
    def fit(features: pd.DataFrame) -> StandardScaler:
        mean = features.mean()
        scale = features.std(ddof=0)
        # Признак с нулевой дисперсией не масштабируется (защита от NaN).
        scale = scale.replace(0.0, 1.0)
        return StandardScaler(mean=mean, scale=scale)


@dataclass(frozen=True)
class OLSFit:
    """Результат обучения OLS на одном обучающем окне."""

    coefficients: pd.Series
    intercept: float
    scaler: StandardScaler
    train_rows: int

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        scaled = self.scaler.transform(features)
        matrix = scaled[self.coefficients.index].to_numpy()
        return self.intercept + matrix @ self.coefficients.to_numpy()


def fit_ols(train: pd.DataFrame, *, feature_columns: list[str]) -> OLSFit:
    """Обучить OLS на обучающих данных ``train`` (содержит целевую колонку)."""
    _require_columns(train, feature_columns + [TARGET_COLUMN])
    if len(train) == 0:
        raise FitError("обучающее окно пусто")
    if len(train) < 2:
        raise FitError(f"обучающее окно содержит {len(train)} строк; нужна хотя бы 2")

    features = train[feature_columns].astype(float)
    target = train[TARGET_COLUMN].astype(float)
    scaler = StandardScaler.fit(features)
    scaled = scaler.transform(features)
    matrix = scaled.to_numpy()
    design = np.column_stack([np.ones(len(matrix)), matrix])

    # OLS через решение системы нормальных уравнений (чистый numpy).
    try:
        coef_all, *_ = np.linalg.lstsq(design, target.to_numpy(), rcond=None)
    except np.linalg.LinAlgError as exc:  # pragma: no cover
        raise FitError(f"не удалось решить систему нормальных уравнений: {exc}") from exc

    intercept = float(coef_all[0])
    coefficients = pd.Series(coef_all[1:], index=feature_columns)

    # Отбрасываем неинформативные признаки (нулевой коэффициент-вклад) не будем —
    # сохраняем полный вектор, чтобы маска наблюдений оставалась единой.
    return OLSFit(
        coefficients=coefficients.astype(float),
        intercept=intercept,
        scaler=scaler,
        train_rows=len(train),
    )


@dataclass(frozen=True)
class Forecast:
    """Прогноз на целевой месяц t+1 и фактические значения."""

    forecast_month: date
    predicted: pd.Series
    actual: pd.Series
    train_rows: int


def walk_forward(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    training_window_months: int,
    refit_frequency: str,
    first_target_month: date,
    last_month: date,
) -> list[Forecast]:
    """Скользящее переобучение и прогноз по каждому целевому месяцу.

    Прогноз на месяц ``m`` строится на обучающих наблюдениях с целевыми
    месяцами в окне ``[m - training_window_months - 1, m - 1]`` (пары s из
    [t−W, t−1] при t+1 = m). При ``refit_frequency = "monthly"`` модель
    переобучается в каждом целевом месяце; поддерживается только
    ежемесячное переобучение (остальные значения не реализованы).
    """
    if refit_frequency != "monthly":
        raise FitError(
            f"неподдерживаемая частота переобучения: {refit_frequency!r} "
            "(ожидается 'monthly')"
        )

    frame = frame.copy()
    frame["month"] = pd.to_datetime(frame["month"])
    forecasts: list[Forecast] = []

    target_months = sorted(set(frame["month"]))
    for m in target_months:
        if m < pd.Timestamp(first_target_month):
            continue
        if m > pd.Timestamp(last_month):
            break

        # Окно: месяцы пар s от (m - W - 1) до (m - 1).
        window_start = _shift_month(m, -(training_window_months + 1))
        window_end = _shift_month(m, -1)
        window = frame[
            (frame["month"] >= window_start) & (frame["month"] <= window_end)
        ]
        if len(window) < 2:
            continue

        fitted = fit_ols(window, feature_columns=feature_columns)
        current = frame[frame["month"] == m]
        if current.empty:
            continue
        predicted = pd.Series(
            fitted.predict(current[feature_columns]),
            index=current.index,
        )
        forecasts.append(
            Forecast(
                forecast_month=m.date(),
                predicted=predicted,
                actual=current[TARGET_COLUMN].astype(float).reset_index(drop=True),
                train_rows=fitted.train_rows,
            )
        )

    return forecasts


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise FitError(f"отсутствуют колонки: {missing}")


def _shift_month(value: pd.Timestamp, months: int) -> pd.Timestamp:
    total = value.year * 12 + (value.month - 1) + months
    year, month_index = divmod(total, 12)
    return pd.Timestamp(date(year, month_index + 1, 1))