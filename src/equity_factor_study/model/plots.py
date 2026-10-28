"""Модельный слой: графическая визуализация результатов отбора (разделы 4.1–4.2).

Строит диагностические графики методики из документа «Оценка вневыборочной
полезности факторов»:

* динамика месячных ошибок моделей B и C на периоде оценки (раздел 4.1);
* поведение правила отбора: какой состав выбирался на каждой дате пересмотра
  (раздел 4.1, документирование);
* распределение бутстрэп-средних разности MSE C−B с границами интервала и
  линией нуля (раздел 4.2).

Модуль не обращается к сети. Графики сохраняются в PNG-файлы.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без графического интерфейса (headless)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .bootstrap import BootstrapResult
from .selection import ModelComparison


def plot_monthly_errors(
    comparison: ModelComparison,
    *,
    path: Path,
    title: str = "Динамика месячной MSE моделей B и C (период оценки)",
) -> Path:
    """График месячной MSE моделей B и C на периоде оценки (раздел 4.1).

    Диагностическая функция: позволяет увидеть, различаются ли модели
    систематически на протяжении периода оценки или расхождение сосредоточено
    в отдельных интервалах.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(comparison.monthly_mse_b.index, comparison.monthly_mse_b.to_numpy(),
            label="B (полный набор)", color="#1f77b4", linewidth=1.6)
    ax.plot(comparison.monthly_mse_c.index, comparison.monthly_mse_c.to_numpy(),
            label="C (обновляемый состав)", color="#d62728", linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("Целевой месяц")
    ax.set_ylabel("Месячная MSE (агрегация по акциям)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_rebalance_history(
    comparison: ModelComparison,
    *,
    path: Path,
    title: str = "Выбранный состав факторов на датах пересмотра",
) -> Path:
    """График поведения правила отбора состава (раздел 4.1).

    Число признаков выбранного состава на каждой дате пересмотра. Частая смена
    состава при отсутствии выигрыша в точности свидетельствует о реакции на
    шум, а неизменный состав — о том, что правило не активировалось.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hist = comparison.rebalance_history
    fig, ax = plt.subplots(figsize=(10, 4))
    if hist.empty:
        ax.text(0.5, 0.5, "Нет дат пересмотра", ha="center", va="center",
                transform=ax.transAxes)
    else:
        dates = pd.to_datetime(hist["rebalance_date"])
        # Число признаков из имени состава.
        sizes = [_compose_size(name) for name in hist["selected_groups"]]
        ax.step(dates, sizes, where="post", color="#2ca02c", linewidth=1.8)
        ax.set_ylabel("Число признаков выбранного состава")
        ax.set_ylim(-0.5, 5.5)
        ax.grid(alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("Дата пересмотра")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_bootstrap_distribution(
    boot: BootstrapResult,
    *,
    path: Path,
    title: str = "Бутстрэп-распределение средней разности MSE (C−B)",
) -> Path:
    """Гистограмма распределения бутстрэп-средних разности C−B (раздел 4.2).

    Отрисываются границы доверительного интервала и линия нуля. Если интервал
    целиком выше нуля — выигрыш от обновления состава; ниже нуля — ухудшение;
    содержит ноль — данных недостаточно.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not boot.means:
        raise ValueError("BootstrapResult не содержит бутстрэп-средних (means)")
    means = np.asarray(boot.means, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(means, bins=60, color="#4472c4", alpha=0.75, label="Бутстрэп-средние")
    ax.axvline(boot.point_estimate, color="black", linestyle="--", linewidth=1.8,
               label=f"Точечная оценка: {boot.point_estimate:.4f}")
    ax.axvline(boot.lower, color="orangered", linestyle=":", linewidth=1.6,
               label=f"Нижняя граница: {boot.lower:.4f}")
    ax.axvline(boot.upper, color="orangered", linestyle=":", linewidth=1.6,
               label=f"Верхняя граница: {boot.upper:.4f}")
    ax.axvline(0.0, color="green", linewidth=1.4, label="0")
    ax.set_xlabel("Средняя разность MSE (C−B)")
    ax.set_ylabel("Частота")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# Соответствие имени группы числу признаков состава (для графика пересмотра).
_GROUP_NAMES_TO_SIZE = {
    "динамика": 2,
    "риск": 1,
    "макро": 2,
}


def _compose_size(name: str) -> int:
    if not name or name == "пустой":
        return 0
    parts = [p for p in name.split("+") if p]
    return sum(_GROUP_NAMES_TO_SIZE.get(p, 0) for p in parts)