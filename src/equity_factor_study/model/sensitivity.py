"""Модельный слой: проверки устойчивости процедуры отбора (раздел 4.3).

Реализует проверки методики из документа «Оценка вневыборочной полезности
факторов» (раздел 4.3). Каждая проверка изменяет один элемент конструкции при
неизменности остальных, позволяя атрибутировать изменение результата
конкретному параметру:

* длина обучающего окна (36 / 48 / 60 мес.);
* длина проверочного периода при отборе состава (12 / 18 мес.);
* длина блока бутстрэпа (3 / 6 / 12 мес.);
* состав выборки (последовательное исключение отдельных бумаг);
* чувствительность к экстремальным месяцам (исключение по квантилю ошибок);
* основная метрика (MSE / MAE).

Итог — сводная таблица, где по строкам альтернативные спецификации, а по
столбцам средняя разность ошибок моделей B и C и границы её доверительного
интервала. Модуль не обращается к сети и не изменяет исходные данные.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .bootstrap import BootstrapResult, paired_block_bootstrap
from .selection import (
    ModelComparison,
    run_selection_evaluation,
)

# Длина основного обучающего окна и проверочного периода — из конфигурации,
# передаются вызывающей стороной; здесь только определения вспомогательных
# обёрток и таблица результатов.


@dataclass(frozen=True)
class SensitivityRow:
    """Результат одной спецификации проверки устойчивости."""

    specification: str           # название спецификации
    comparison_ba: float         # средняя разность ошибок (C−B)
    lower: float                 # нижняя граница интервала
    upper: float                 # верхняя граница интервала
    ci_contains_zero: bool       # содержит ли интервал ноль
    n_observations: int          # число наблюдений ряда разностей
    detail: str = ""             # доп. сведения (тип исключения и т.п.)


def _bootstrap_ci(diff: pd.Series, *, block_months: int, replicates: int,
                  confidence: float, seed: int) -> BootstrapResult:
    return paired_block_bootstrap(
        diff, block_months=block_months, replicates=replicates,
        confidence=confidence, seed=seed,
    )


def _diff_series(comparison: ModelComparison) -> pd.Series:
    """Ряд месячных разностей (C−B) MSE на периоде оценки."""
    return (comparison.monthly_mse_c - comparison.monthly_mse_b).dropna()


def sensitivity_by_window(
    frame: pd.DataFrame,
    *,
    windows: list[int],
    first_target_month: date,
    evaluation_target_start: date,
    last_month: date,
    selection_window_months: int,
    block_months: int,
    replicates: int,
    confidence: float,
    seed: int,
) -> list[SensitivityRow]:
    """Проверка чувствительности к длине обучающего окна (36/48/60)."""
    rows: list[SensitivityRow] = []
    for window in windows:
        comparison = run_selection_evaluation(
            frame,
            training_window_months=window,
            first_target_month=first_target_month,
            evaluation_target_start=evaluation_target_start,
            last_month=last_month,
            selection_window_months=selection_window_months,
            per_security=True,
        )
        diff = _diff_series(comparison)
        if diff.empty:
            continue
        boot = _bootstrap_ci(
            diff, block_months=block_months, replicates=replicates,
            confidence=confidence, seed=seed,
        )
        rows.append(_row(
            f"окно={window} мес.", comparison.comparison_ba, boot,
        ))
    return rows


def sensitivity_by_selection_window(
    frame: pd.DataFrame,
    *,
    training_window_months: int,
    selection_windows: list[int],
    first_target_month: date,
    evaluation_target_start: date,
    last_month: date,
    block_months: int,
    replicates: int,
    confidence: float,
    seed: int,
) -> list[SensitivityRow]:
    """Проверка чувствительности к длине проверочного периода отбора (12/18)."""
    rows: list[SensitivityRow] = []
    for sel_window in selection_windows:
        comparison = run_selection_evaluation(
            frame,
            training_window_months=training_window_months,
            first_target_month=first_target_month,
            evaluation_target_start=evaluation_target_start,
            last_month=last_month,
            selection_window_months=sel_window,
            per_security=True,
        )
        diff = _diff_series(comparison)
        if diff.empty:
            continue
        boot = _bootstrap_ci(
            diff, block_months=block_months, replicates=replicates,
            confidence=confidence, seed=seed,
        )
        rows.append(_row(
            f"выбор по {sel_window} мес.", comparison.comparison_ba, boot,
        ))
    return rows


def sensitivity_by_block(
    comparison: ModelComparison,
    *,
    blocks: list[int],
    replicates: int,
    confidence: float,
    seed: int,
) -> list[SensitivityRow]:
    """Проверка чувствительности к длине блока бутстрэпа (3/6/12).

    Не требует пересчёта моделей: меняется только бутстрэп на фиксированном
    ряду разностей. Точечная оценка одинакова, различается лишь ширина
    интервала.
    """
    diff = _diff_series(comparison)
    rows: list[SensitivityRow] = []
    for block in blocks:
        boot = _bootstrap_ci(
            diff, block_months=block, replicates=replicates,
            confidence=confidence, seed=seed,
        )
        rows.append(_row(
            f"блок={block} мес.", comparison.comparison_ba, boot,
        ))
    return rows


def sensitivity_by_exclusion(
    frame: pd.DataFrame,
    *,
    training_window_months: int,
    first_target_month: date,
    evaluation_target_start: date,
    last_month: date,
    selection_window_months: int,
    block_months: int,
    replicates: int,
    confidence: float,
    seed: int,
) -> list[SensitivityRow]:
    """Проверка чувствительности к составу выборки: исключение каждой бумаги.

    Для каждого инструмента из ``frame`` его наблюдения исключаются, и
    процедура отбора пересчитывается. Позволяет увидеть, определяется ли
    результат поведением одного-двух инструментов.
    """
    base = _frame(
        frame,
        training_window_months=training_window_months,
        first_target_month=first_target_month,
        evaluation_target_start=evaluation_target_start,
        last_month=last_month,
        selection_window_months=selection_window_months,
        block_months=block_months,
        replicates=replicates,
        confidence=confidence,
        seed=seed,
    )
    rows: list[SensitivityRow] = [base]
    secids = sorted(set(frame["secid"]))
    for secid in secids:
        reduced = frame[frame["secid"] != secid]
        comparison = run_selection_evaluation(
            reduced,
            training_window_months=training_window_months,
            first_target_month=first_target_month,
            evaluation_target_start=evaluation_target_start,
            last_month=last_month,
            selection_window_months=selection_window_months,
            per_security=True,
        )
        diff = _diff_series(comparison)
        if diff.empty:
            continue
        boot = _bootstrap_ci(
            diff, block_months=block_months, replicates=replicates,
            confidence=confidence, seed=seed,
        )
        rows.append(_row(
            f"без {secid}", comparison.comparison_ba, boot, detail=secid,
        ))
    return rows


def sensitivity_by_extreme_months(
    comparison: ModelComparison,
    *,
    months: list[int] | None = None,
    block_months: int,
    replicates: int,
    confidence: float,
    seed: int,
) -> list[SensitivityRow]:
    """Чувствительность к экстремальным месяцам: исключение по квантилю ошибок.

    Исключаются месяцы, в которых ошибка прогноза превышает заданный квантиль
    распределения ошибок (например, 80-й и 90-й перцентиль ряда MSE модели B).
    По умолчанию ``months`` = [80, 90].
    """
    months = months or [80, 90]
    diff = _diff_series(comparison)
    mse_b = comparison.monthly_mse_b.dropna()
    rows: list[SensitivityRow] = []
    for q in months:
        threshold = np.quantile(mse_b, q / 100.0) if len(mse_b) else float("inf")
        keep = mse_b[mse_b.index.isin(diff.index)]
        retained = keep[keep <= threshold].index
        filtered = diff[diff.index.isin(retained)]
        if filtered.empty:
            continue
        boot = _bootstrap_ci(
            filtered, block_months=block_months, replicates=replicates,
            confidence=confidence, seed=seed,
        )
        rows.append(_row(
            f"без месяцев с ошибкой > {q}-го квант.",
            float(filtered.mean()), boot, detail=f"исключено {len(diff) - len(filtered)}",
        ))
    return rows


def sensitivity_by_metric(
    comparison: ModelComparison,
    *,
    block_months: int,
    replicates: int,
    confidence: float,
    seed: int,
) -> list[SensitivityRow]:
    """Чувствительность к основной метрике: MAE вместо MSE.

    Разность строится по месячным MAE (C−B) и бутстрэпится так же, как для MSE.
    Совпадение знака с разностью по MSE указывает, что результат не определяется
    небольшим числом крупных отклонений.
    """
    diff = (comparison.monthly_mae_c - comparison.monthly_mae_b).dropna()
    rows: list[SensitivityRow] = []
    if diff.empty:
        return rows
    boot = _bootstrap_ci(
        diff, block_months=block_months, replicates=replicates,
        confidence=confidence, seed=seed,
    )
    rows.append(_row("метрика MAE", float(diff.mean()), boot))
    return rows


def _frame(
    frame: pd.DataFrame,
    *,
    training_window_months: int,
    first_target_month: date,
    evaluation_target_start: date,
    last_month: date,
    selection_window_months: int,
    block_months: int,
    replicates: int,
    confidence: float,
    seed: int,
) -> SensitivityRow:
    """Основная (эталонная) спецификация для сравнения с альтернативами."""
    comparison = run_selection_evaluation(
        frame,
        training_window_months=training_window_months,
        first_target_month=first_target_month,
        evaluation_target_start=evaluation_target_start,
        last_month=last_month,
        selection_window_months=selection_window_months,
        per_security=True,
    )
    diff = _diff_series(comparison)
    boot = _bootstrap_ci(
        diff, block_months=block_months, replicates=replicates,
        confidence=confidence, seed=seed,
    )
    return _row("базовая (окно=48, выбор=12, блок=6)", comparison.comparison_ba, boot)


def _row(specification: str, comparison_ba: float, boot: BootstrapResult, detail: str = "") -> SensitivityRow:
    return SensitivityRow(
        specification=specification,
        comparison_ba=comparison_ba,
        lower=boot.lower,
        upper=boot.upper,
        ci_contains_zero=(boot.lower <= 0 <= boot.upper),
        n_observations=boot.n_observations,
        detail=detail,
    )


def sensitivity_summary(rows: list[SensitivityRow]) -> pd.DataFrame:
    """Свести результаты проверок устойчивости в единую таблицу."""
    if not rows:
        return pd.DataFrame(
            columns=["specification", "comparison_ba", "lower", "upper",
                     "ci_contains_zero", "n_observations", "detail"]
        )
    return pd.DataFrame([vars(r) for r in rows])