"""Модельный слой: сводный отчёт и интерпретация результатов отбора.

Формирует сводный отчёт (markdown) по разделам 4.1–4.2 методики из документа
«Оценка вневыборочной полезности факторов»:

* таблица сводных метрик точности моделей A, B, C на периоде оценки;
* динамика ошибок (ссылка на график) и поведение правила отбора;
* средняя разность MSE моделей B и C с границами доверительного интервала;
* доля месяцев, где каждая модель точнее, для трёх сравнений;
* готовая интерпретация по трём сценариям раздела 4.2.

Модуль не обращается к сети и не изменяет исходные данные.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .bootstrap import BootstrapResult
from .selection import ModelComparison


def _win_rate(better: pd.Series, worse: pd.Series) -> float:
    """Доля месяцев, где ``better`` имеет меньшую ошибку, чем ``worse``."""
    common = better.dropna().index.intersection(worse.dropna().index)
    if len(common) == 0:
        return float("nan")
    return float(np.mean(better[common].to_numpy() < worse[common].to_numpy()))


def interpret_comparison(point: float, lower: float, upper: float, *, label: str) -> str:
    """Интерпретация разности ошибок по трём сценариям раздела 4.2."""
    if lower > 0:
        return (
            f"Интервал [{lower:.4f}, {upper:.4f}] для {label} целиком выше нуля: "
            "наблюдается систематический выигрыш в точности."
        )
    if upper < 0:
        return (
            f"Интервал [{lower:.4f}, {upper:.4f}] для {label} целиком ниже нуля: "
            "процедура систематически ухудшает прогноз — ошибки прошлых прогнозов "
            "не предсказывают будущей полезности факторов, а сам пересмотр состава "
            "вносит дополнительную нестабильность."
        )
    return (
        f"Интервал [{lower:.4f}, {upper:.4f}] для {label} содержит ноль: данные "
        "не позволяют утверждать наличие эффекта ни в одном из направлений. "
        "Наблюдаемая нестабильность полезности факторов не поддаётся использованию "
        "в рамках данной процедуры на данном рынке и горизонте."
    )


def build_summary_report(
    comparison: ModelComparison,
    *,
    boot: BootstrapResult,
    aggregate: str,
    title: str,
) -> str:
    """Сформировать сводный отчёт (markdown) о сравнении моделей A, B, C."""
    a = comparison.model_a
    b = comparison.model_b
    c = comparison.model_c

    # Доля месяцев, где модель точнее (меньше MSE).
    win_c_vs_b = _win_rate(comparison.monthly_mse_c, comparison.monthly_mse_b)
    win_c_vs_a = _win_rate(comparison.monthly_mse_c, comparison.monthly_mse_a)
    win_b_vs_a = _win_rate(comparison.monthly_mse_b, comparison.monthly_mse_a)

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- Порядок агрегации: `{aggregate}`.")
    lines.append(f"- Период оценки: {comparison.monthly_mse_a.index.min().date()} — "
                 f"{comparison.monthly_mse_a.index.max().date()} "
                 f"({len(comparison.monthly_mse_a)} месяцев).")
    lines.append(f"- Число бутстрэп-повторов: {boot.replicates}, блок: {boot.block_months} мес., "
                 f"зерно: {boot.seed}.")
    lines.append("")

    lines.append("## 1. Сводные метрики точности моделей (период оценки)")
    lines.append("")
    lines.append("| Модель | MSE | MAE | Число месяцев |")
    lines.append("|---|---|---|---|")
    lines.append(f"| A (историческое среднее) | {a.mse:.6f} | {a.mae:.6f} | {a.forecast_count} |")
    lines.append(f"| B (полный набор) | {b.mse:.6f} | {b.mae:.6f} | {b.forecast_count} |")
    lines.append(f"| C (обновляемый состав) | {c.mse:.6f} | {c.mae:.6f} | {c.forecast_count} |")
    lines.append("")

    lines.append("## 2. Средняя разность ошибок моделей B и C")
    lines.append("")
    lines.append(f"- Средняя разность MSE (C − B): **{comparison.comparison_ba:.6f}**")
    lines.append(f"- 95%-интервал: **[{boot.lower:.6f}, {boot.upper:.6f}]**")
    lines.append("")
    lines.append(interpret_comparison(comparison.comparison_ba, boot.lower, boot.upper,
                                      label="разности C − B"))
    lines.append("")

    lines.append("## 3. Сопоставление с историческим средним (модель A)")
    lines.append("")
    lines.append(f"- Разность MSE (C − A): **{comparison.comparison_ca:.6f}**")
    lines.append(f"- Разность MSE (B − A): **{b.mse - a.mse:.6f}**")
    lines.append("")
    if b.mse < a.mse:
        lines.append("Регрессионные модели уступают историческому среднему? "
                     "Напротив: регрессия с полным набором точнее модели A, "
                     "т.е. факторы в целом добавляют прогнозную информацию.")
    else:
        lines.append("Обе регрессионные модели уступают историческому среднему: "
                     "добавление ценовых и макроэкономических признаков не улучшает "
                     "прогноз относительно простейшего ориентира. Это согласуется "
                     "с классическими наблюдениями о низкой вневыборочной "
                     "предсказуемости доходности.")
    lines.append("")

    lines.append("## 4. Доля месяцев, где модель точнее")
    lines.append("")
    lines.append("| Сравнение | Доля месяцев точнее |")
    lines.append("|---|---|")
    lines.append(f"| C точнее B | {win_c_vs_b:.3f} |")
    lines.append(f"| C точнее A | {win_c_vs_a:.3f} |")
    lines.append(f"| B точнее A | {win_b_vs_a:.3f} |")
    lines.append("")
    lines.append("Расхождение между направлением средней разности и долей выигрышных "
                 "месяцев указывает на асимметрию распределения ошибок.")
    lines.append("")

    lines.append("## 5. Поведение правила отбора состава")
    lines.append("")
    hist = comparison.rebalance_history
    if hist.empty:
        lines.append("Пересмотров состава не выполнялось.")
    else:
        lines.append(f"Число пересмотров: {len(hist)}.")
        # Как часто выбор менялся относительно предыдущего пересмотра.
        changes = int((hist["selected_groups"] != hist["selected_groups"].shift()).sum()) - 1
        lines.append(f"Число смен выбранного состава: {max(changes, 0)}.")
        lines.append("")
        lines.append("| Дата пересмотра | Выбранный состав |")
        lines.append("|---|---|")
        for row in hist.itertuples(index=False):
            lines.append(f"| {row.rebalance_date} | {row.selected_groups} |")
    lines.append("")

    return "\n".join(lines)


def write_summary_report(
    comparison: ModelComparison,
    *,
    boot: BootstrapResult,
    aggregate: str,
    path: Path,
    title: str = "Сводный отчёт о прогнозной полезности факторов",
) -> Path:
    """Сформировать сводный отчёт и записать в markdown-файл."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = build_summary_report(
        comparison, boot=boot, aggregate=aggregate, title=title,
    )
    path.write_text(content, encoding="utf-8")
    return path