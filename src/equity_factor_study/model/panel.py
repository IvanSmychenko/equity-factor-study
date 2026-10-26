"""Модельный слой: формирование матрицы наблюдений из месячной панели.

Этот модуль — отдельный этап анализа поверх готовой панели
(`monthly_panel.csv`), не входящий в этапы 1–8 ТЗ (границы работ, §6). Он
соблюдает инварианты протокола: единая маска доступности (инвариант 5) и
отсутствие утечки (инвариант 1).

Наблюдение — пара «инструмент — месяц» с полным набором признаков и целевой
переменной. Маска доступности строится **один раз** и используется одинаково
для всех наборов признаков: модель с сокращённым набором признаков не
получает доступа к большему числу наблюдений, чем модель с полным набором.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

# Порядок и имена признаков панели (совпадают со схемой data_dictionary.md).
FEATURE_COLUMNS = [
    "return_1m",
    "momentum_prev_3m",
    "volatility_3m",
    "key_rate_pct",
    "key_rate_change_pp",
]
TARGET_COLUMN = "target_return_next_1m"
IDENTITY_COLUMNS = ["secid", "month"]


class PanelDataError(Exception):
    """Ошибка формирования матрицы наблюдений из панели."""


@dataclass(frozen=True)
class ObservationMatrix:
    """Матрица наблюдений: признаки, целевая переменная и метаданные строки.

    Строки — наблюдения, прошедшие единую маску доступности. ``features`` —
    DataFrame колонок признаков (без целевой), ``target`` — Series целевой
    переменной, ``identity`` — DataFrame с secid и month для идентификации
    наблюдения (для среза разработки/оценки и проверки маски).
    """

    features: pd.DataFrame
    target: pd.Series
    identity: pd.DataFrame


def _check_required_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise PanelDataError(f"{label}: отсутствуют колонки {missing}")


def build_observation_matrix(panel: pd.DataFrame) -> ObservationMatrix:
    """Построить матрицу наблюдений по единой маске доступности.

    Наблюдение пригодно, если присутствуют все пять признаков и непустая
    целевая переменная. Пропуски, помеченные в ``quality_flags`` (например,
    отсутствие торгов), исключаются, а не заполняются.
    """
    _check_required_columns(
        panel, IDENTITY_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN, "quality_flags"],
        "панель",
    )

    mask = panel[FEATURE_COLUMNS].notna().all(axis=1) & panel[TARGET_COLUMN].notna()
    if not mask.any():
        raise PanelDataError("панель не содержит наблюдений с полным набором признаков")

    usable = panel[mask].copy()
    features = usable[FEATURE_COLUMNS].reset_index(drop=True)
    target = usable[TARGET_COLUMN].astype(float).reset_index(drop=True)
    identity = usable[IDENTITY_COLUMNS].reset_index(drop=True)
    return ObservationMatrix(
        features=features.astype(float),
        target=target,
        identity=identity,
    )


def select_features(matrix: ObservationMatrix, columns: list[str]) -> ObservationMatrix:
    """Вернуть матрицу с подмножеством признаков.

    Строки (наблюдения) остаются теми же, что у полного набора, — это и есть
    соблюдение единой маски доступности для сокращённого набора признаков
    (инвариант 5).
    """
    missing = [c for c in columns if c not in matrix.features.columns]
    if missing:
        raise PanelDataError(f"запрошены отсутствующие признаки: {missing}")
    return ObservationMatrix(
        features=matrix.features[columns].copy(),
        target=matrix.target,
        identity=matrix.identity.copy(),
    )


def split_by_period(
    matrix: ObservationMatrix,
    *,
    development_target_end: date,
    evaluation_target_start: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Разделить наблюдения на период разработки и период оценки.

    Классификация по целевому месяцу ``month``:
    * разработка — месяцы до ``development_target_end`` включительно;
    * оценка — месяцы от ``evaluation_target_start`` включительно.
    Возвращает кортеж (матрица разработки, матрица оценки) с теми же колонками,
    что и ``matrix.features`` + целевая.
    """
    months = pd.to_datetime(matrix.identity["month"])
    dev_mask = months <= pd.Timestamp(development_target_end)
    dev = _frame_with_target(matrix, dev_mask)
    ev_mask = months >= pd.Timestamp(evaluation_target_start)
    ev = _frame_with_target(matrix, ev_mask)
    return dev, ev


def _frame_with_target(matrix: ObservationMatrix, mask: pd.Series) -> pd.DataFrame:
    out = matrix.features.loc[mask.to_numpy()].copy()
    out[TARGET_COLUMN] = matrix.target.loc[mask.to_numpy()].to_numpy()
    out["secid"] = matrix.identity["secid"].loc[mask.to_numpy()].to_numpy()
    out["month"] = matrix.identity["month"].loc[mask.to_numpy()].to_numpy()
    return out.reset_index(drop=True)