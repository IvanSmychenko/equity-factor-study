"""Модельный слой: парный блочный бутстрэп для разности ошибок (модель C−B).

Реализует методику из документа «Оценка вневыборочной полезности факторов
в моделях доходности российских акций» (раздел 3.5). Парность означает, что
пересэмплированию подвергается ряд месячных разностей ошибок моделей B и C,
вычисленных на одних и тех же месяцах. Блочная структура сохраняет временную
зависимость (кластеры крупных ошибок).

Модуль соблюдает инварианты протокола: детерминизм (фиксированное зерно),
отсутствие утечки и отсутствие обращения к сети.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class BootstrapError(Exception):
    """Ошибка бутстрэп-процедуры."""


@dataclass(frozen=True)
class BootstrapResult:
    """Результат бутстрэпа разности ошибок Y=A; C−B."""

    point_estimate: float          # средняя разность (точечная оценка)
    mean: float                    # среднее по псевдовыборкам
    lower: float                   # нижняя граница доверительного интервала
    upper: float                   # верхняя граница доверительного интервала
    replicates: int                # число повторов
    block_months: int              # длина блока
    confidence: float              # доверительный уровень
    seed: int                      # зерно генератора
    n_blocks: int                  # число блоков (информация для воспроизводимости)
    n_observations: int            # длина исходного ряда
    means: tuple[float, ...] = ()  # бутстрэп-средние по повторам (для графиков)


def paired_block_bootstrap(
    monthly_difference: pd.Series,
    *,
    block_months: int = 6,
    replicates: int = 5000,
    confidence: float = 0.95,
    seed: int = 20140101,
) -> BootstrapResult:
    """Парный блочный бутстрэп по месячному ряду разностей ошибок.

    ``monthly_difference`` — ряд месячных разностей (C−B) MSE, индекс — первое
    число месяца. Ряд сортируется по индексу; блоки берутся перекрывающимися
    длиной ``block_months``. На каждой псевдовыборке блоки извлекаются с
    возвращением до достижения исходной длины ряда; средняя разность на
    псевдовыборке — это среднее значение выбранных наблюдений. По эмпирическому
    распределению средних строятся границы ``confidence``-интервала.

    Процедура детерминирована: ``seed`` фиксирует генератор.
    """
    series = monthly_difference.sort_index().dropna()
    if series.empty:
        raise BootstrapError("ряд месячных разностей пуст")
    if block_months <= 0:
        raise BootstrapError("длина блока должна быть положительным целым")
    if block_months > len(series):
        raise BootstrapError(
            f"длина блока {block_months} больше длины ряда {len(series)}"
        )
    if replicates <= 0:
        raise BootstrapError("число повторов должно быть положительным целым")
    if not 0.0 < confidence < 1.0:
        raise BootstrapError("доверительный уровень должен лежать в интервале (0, 1)")

    values = series.to_numpy(dtype=float)
    n = len(values)
    point = float(values.mean())

    # Перекрывающиеся блоки длины block_months.
    n_blocks = n - block_months + 1
    blocks = np.lib.stride_tricks.sliding_window_view(values, block_months)

    rng = np.random.default_rng(seed)
    # Число блоков, которые нужно извлечь, чтобы покрыть исходную длину.
    n_draw = int(np.ceil(n / block_months))
    means = np.empty(replicates, dtype=float)
    for i in range(replicates):
        idx = rng.integers(0, n_blocks, size=n_draw)
        sample = blocks[idx].reshape(-1)[:n]
        means[i] = sample.mean()

    boot_mean = float(means.mean())
    alpha = 1.0 - confidence
    lower, upper = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])

    return BootstrapResult(
        point_estimate=point,
        mean=boot_mean,
        lower=float(lower),
        upper=float(upper),
        replicates=replicates,
        block_months=block_months,
        confidence=confidence,
        seed=seed,
        n_blocks=n_blocks,
        n_observations=n,
        means=tuple(means.tolist()),
    )


def bootstrap_for_comparison(
    monthly_error_b: pd.Series,
    monthly_error_c: pd.Series,
    *,
    block_months: int = 6,
    replicates: int = 5000,
    confidence: float = 0.95,
    seed: int = 20140101,
) -> BootstrapResult:
    """Парный бутстрэп разности MSE (C−B) на общем множестве месяцев.

    Обе модели проецируются на пересечение целевых месяцев (единая маска
    наблюдений); ряд разностей C−B строится только там, где есть оба значения.
    """
    diff = monthly_error_c - monthly_error_b
    return paired_block_bootstrap(
        diff,
        block_months=block_months,
        replicates=replicates,
        confidence=confidence,
        seed=seed,
    )