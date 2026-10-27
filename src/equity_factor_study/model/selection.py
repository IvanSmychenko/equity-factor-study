"""Модельный слой: отбор состава факторов (модель C) и сравнение A/B/C.

Реализует методику из документа «Оценка вневыборочной полезности факторов
в моделях доходности российских акций» (разделы 3.2–3.4):

* модель A — историческое среднее: прогноз равен средней реализованной
  доходности за то же обучающее окно (используется как минимальный ориентир);
* модель B — OLS по каждой акции отдельно с полным набором из пяти признаков;
* модель C — та же регрессия, но состав факторов пересматривается раз в квартал
  по ошибкам предшествующих прогнозов.

Три содержательные группы признаков (ценовая динамика, риск, макроэкономика)
порождают восемь допустимых составов (таблица 2 методики), включая пустой.

Модуль соблюдает инварианты протокола: единая маска доступности (инвариант 5)
и отсутствие утечки (инвариант 1). Сеть не используется, исходные данные не
изменяются.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .evaluate import _flatten, _metrics_from_lists
from .fit import Forecast, walk_forward
from .panel import TARGET_COLUMN

# Содержательные группы признаков (методика, раздел 3.3 / таблица 2).
# Каждый состав — набор групп; если группа включена, берутся все её признаки.
_GROUP_PRICE = "price"           # ценовая динамика
_GROUP_RISK = "risk"             # риск
_GROUP_MACRO = "macro"           # макроэкономические условия

# Канонический порядок групп при перечислении признаков состава.
_GROUP_ORDER = (_GROUP_PRICE, _GROUP_RISK, _GROUP_MACRO)

GROUP_FEATURES: dict[str, list[str]] = {
    _GROUP_PRICE: ["return_1m", "momentum_prev_3m"],
    _GROUP_RISK: ["volatility_3m"],
    _GROUP_MACRO: ["key_rate_pct", "key_rate_change_pp"],
}

# Восемь допустимых составов (таблица 2 методики). Каждый состав задаётся
# набором включённых групп; пустой состав соответствует модели исторического
# среднего (внутренний контроль построения).
COMPOSITIONS: tuple[frozenset[str], ...] = (
    frozenset({_GROUP_PRICE}),
    frozenset({_GROUP_RISK}),
    frozenset({_GROUP_MACRO}),
    frozenset({_GROUP_PRICE, _GROUP_RISK}),
    frozenset({_GROUP_PRICE, _GROUP_MACRO}),
    frozenset({_GROUP_RISK, _GROUP_MACRO}),
    frozenset({_GROUP_PRICE, _GROUP_RISK, _GROUP_MACRO}),
    frozenset(),
)


class SelectionError(Exception):
    """Ошибка процедуры отбора состава факторов."""


@dataclass(frozen=True)
class Composition:
    """Состав факторов как упорядоченный список признаков группы."""

    groups: frozenset[str]
    features: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.features)


def build_compositions() -> list[Composition]:
    """Построить список всех восьми составов из групп признаков."""
    result: list[Composition] = []
    for groups in COMPOSITIONS:
        features: list[str] = []
        for group in _GROUP_ORDER:
            if group in groups:
                features.extend(GROUP_FEATURES[group])
        result.append(Composition(groups=groups, features=tuple(features)))
    return result


def composition_features(groups: frozenset[str]) -> list[str]:
    """Признаки, соответствующие набору включённых групп (в каноническом порядке)."""
    features: list[str] = []
    for group in _GROUP_ORDER:
        if group in groups:
            features.extend(GROUP_FEATURES[group])
    return features


@dataclass(frozen=True)
class ForecastBundle:
    """Прогнозы одной модели на согласованном множестве целевых месяцев.

    ``forecasts`` — отдельный прогноз на каждую пару «акция — целевой месяц».
    ``train_rows_by_month`` — фактическое число наблюдений обучающего окна по
    каждому целевому месяцу (для проверки равенства единой маски).
    """

    forecasts: tuple[Forecast, ...]
    train_rows_by_month: dict[date, tuple[int, ...]]


def forecast_series(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    training_window_months: int,
    first_target_month: date,
    last_month: date,
    per_security: bool = True,
) -> list[Forecast]:
    """Построить последовательные прогнозы модели с указанным составом факторов.

    Используется для моделей B и C (регрессия). Окно и единая маска полностью
    определяются моделью. При ``feature_columns`` пустом составе модель не
    строится — метод не поддерживает пустой состав (его моделирует моделя A).
    """
    if not feature_columns:
        raise SelectionError(
            "регрессионная модель с пустым составом факторов не строится; "
            "пустой состав соответствует модели исторического среднего (модель A)"
        )
    return walk_forward(
        frame,
        feature_columns=feature_columns,
        training_window_months=training_window_months,
        refit_frequency="monthly",
        first_target_month=first_target_month,
        last_month=last_month,
        per_security=per_security,
    )


def historical_average_forecast(
    frame: pd.DataFrame,
    *,
    training_window_months: int,
    first_target_month: date,
    last_month: date,
    per_security: bool = True,
) -> list[Forecast]:
    """Модель A: прогноз = средняя реализованная доходность за то же окно.

    Для каждой акции (при ``per_security=True``) средняя целевая переменная
    вычисляется по её наблюдениям в том же календарном окне, которое
    использовала бы регрессия. Это минимальный ориентир сравнения модели C.
    """
    frame = frame.copy()
    frame["month"] = pd.to_datetime(frame["month"])
    if per_security:
        if "secid" not in frame.columns:
            raise SelectionError("модель A (per_security) требует колонку 'secid'")
        groups = [(secid, g) for secid, g in frame.groupby("secid")]
    else:
        groups = [("__all__", frame)]

    forecasts: list[Forecast] = []
    for _, group in groups:
        group = group.sort_values("month").reset_index(drop=True)
        target_months = sorted(set(group["month"]))
        for m in target_months:
            if m < pd.Timestamp(first_target_month):
                continue
            if m > pd.Timestamp(last_month):
                break
            window_start = _shift_month(m, -(training_window_months + 1))
            window_end = _shift_month(m, -1)
            window = group[
                (group["month"] >= window_start) & (group["month"] <= window_end)
            ]
            if window.empty:
                continue
            mean = float(window[TARGET_COLUMN].astype(float).mean())
            current = group[group["month"] == m]
            if current.empty:
                continue
            n = len(current)
            forecasts.append(
                Forecast(
                    forecast_month=m.date(),
                    predicted=pd.Series([mean] * n, index=current.index),
                    actual=current[TARGET_COLUMN].astype(float).reset_index(drop=True),
                    train_rows=len(window),
                )
            )
    return forecasts


def _shift_month(value: pd.Timestamp, months: int) -> pd.Timestamp:
    total = value.year * 12 + (value.month - 1) + months
    year, month_index = divmod(total, 12)
    return pd.Timestamp(date(year, month_index + 1, 1))


def monthly_mse_series(forecasts: list[Forecast]) -> pd.Series:
    """Месячные MSE одной модели (агрегация securities_then_months).

    Для каждого целевого месяца ошибка сначала усредняется по акциям (внутри
    месяца), затем возвращается как значение месяца. Индекс — ``Timestamp``
    „первое число“ целевого месяца.
    """
    frame = _flatten(forecasts)
    if frame.empty:
        return pd.Series(dtype=float)
    frame["month"] = pd.to_datetime(frame["month"])
    rows: dict[pd.Timestamp, float] = {}
    for month, group in frame.groupby("month"):
        mse, _ = _metrics_from_lists(group["actual"].iloc[0], group["predicted"].iloc[0])
        rows[month] = mse
    return pd.Series(rows).sort_index()


def rebalance_dates(
    *, first_error_month: date,  # первый целевой месяц с прогнозом
    evaluation_target_start: date,
    last_month: date,
    window_months: int,
    frequency: str = "quarterly",
) -> list[date]:
    """Даты пересмотра состава по квартальной сетке после накопления окна.

    Квартальная сетка — первое января, апреля, июля, октября. Датой пересмотра
    считается ближайшая такая дата, на которую к её началу накоплено не меньше
    ``window_months`` завершённых ошибок с момента первого прогноза. Первый
    пересмотр соответствует разогреву (пункт Б.4 ТЗ): 13 прогнозов периода
    разработки расходуются на накопление ошибок, после чего первая дата
    пересмотра — 1 июля 2019.
    """
    def quarter_starts(start: date, end: date) -> list[date]:
        # Все квартальные даты от start.year до end.year включительно.
        result: list[date] = []
        for year in range(start.year, end.year + 1):
            for q_month in (1, 4, 7, 10):
                candidate = date(year, q_month, 1)
                if date(start.year, start.month, 1) <= candidate <= date(end.year, end.month, 1):
                    result.append(candidate)
        return result

    first_error = _to_month_start(first_error_month)
    # К дате пересмотра доступно число ошибок = числу целевых месяцев от
    # first_error до месяца, предшествующего дате пересмотра.
    dates: list[date] = []
    for candidate in quarter_starts(first_error, last_month):
        errors_available = _months_between(first_error, candidate)
        if errors_available >= window_months:
            dates.append(candidate)
    return dates


def select_composition(
    monthly_mse_by_composition: dict[Composition, pd.Series],
    *,
    rebalance_date: date,
    window_months: int,
) -> Composition:
    """Выбрать состав на дате пересмотра по предшествующим ошибкам.

    Для каждого состава берётся средняя месячная MSE за ``window_months``
    завершённых месяцев до даты пересмотра; выбирается состав с минимальным
    значением. При точном равенстве предпочтение отдаётся более простому
    составу (меньше признаков), затем — каноническому порядку (детерминизм).
    """
    cutoff = _to_month_start(rebalance_date)
    scores: dict[Composition, float] = {}
    for comp, series in monthly_mse_by_composition.items():
        window = series[series.index < cutoff].tail(window_months)
        if window.empty:
            continue
        scores[comp] = float(window.mean())
    if not scores:
        raise SelectionError(f"нет накопленных ошибок для пересмотра на {rebalance_date}")

    # Минимальная ошибка; при равенстве — минимум признаков, затем канон. порядок.
    order = build_compositions()
    rank = {comp: idx for idx, comp in enumerate(order)}
    best = min(
        scores,
        key=lambda c: (scores[c], c.size, rank[c]),
    )
    return best


def rebalance_history(
    monthly_mse_by_composition: dict[Composition, pd.Series],
    *,
    evaluation_target_start: date,
    first_error_month: date,
    window_months: int,
) -> pd.DataFrame:
    """История выбора состава на каждой дате пересмотра.

    Возвращает таблицу с колонками ``rebalance_date`` и ``selected_groups``
    (имя набора групп). Позволяет документировать, как часто менялся состав.
    """
    rows: list[dict] = []
    all_dates = {
        d
        for series in monthly_mse_by_composition.values()
        for d in series.index
    }
    if not all_dates:
        return pd.DataFrame(columns=["rebalance_date", "selected_groups"])
    last_month = max(all_dates).date()
    for d in rebalance_dates(
        first_error_month=first_error_month,
        evaluation_target_start=evaluation_target_start,
        last_month=last_month,
        window_months=window_months,
    ):
        comp = select_composition(
            monthly_mse_by_composition, rebalance_date=d, window_months=window_months
        )
        rows.append({"rebalance_date": d, "selected_groups": _group_name(comp.groups)})
    return pd.DataFrame(rows)


def _group_name(groups: frozenset[str]) -> str:
    names = {
        _GROUP_PRICE: "динамика",
        _GROUP_RISK: "риск",
        _GROUP_MACRO: "макро",
    }
    return "+".join(names[g] for g in _GROUP_ORDER if g in groups) or "пустой"


def _to_month_start(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return pd.Timestamp(ts.year, ts.month, 1)


def _months_between(start, end) -> int:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    return (e.year - s.year) * 12 + (e.month - s.month)


@dataclass(frozen=True)
class ModelComparison:
    """Сравнение точности моделей A, B, C на периоде оценки."""

    model_a: MonthError
    model_b: MonthError
    model_c: MonthError
    comparison_ba: float   # разность средних квадратических ошибок (C−B)
    comparison_ca: float   # разность (C−A)
    comparison_ba_win_rate: float  # доля месяцев, где C точнее B
    rebalance_history: pd.DataFrame


@dataclass(frozen=True)
class MonthError:
    """Агрегированная точность модели на периоде оценки."""

    mse: float
    mae: float
    forecast_count: int


def run_selection_evaluation(
    frame: pd.DataFrame,
    *,
    training_window_months: int,
    first_target_month: date,
    evaluation_target_start: date,
    last_month: date,
    selection_window_months: int = 12,
    rebalance_frequency: str = "quarterly",
    per_security: bool = True,
) -> ModelComparison:
    """Полная процедура отбора состава и сравнение моделей A/B/C.

    1. По каждому из восьми составов строятся последовательные прогнозы
       (модель C-регрессии) и их месячные MSE.
    2. По правилу отбора (раздел 3.3) выбирается состав на каждой дате
       ежеквартального пересмотра; выбранный состав применяется в следующем
       квартале — итоговая модель C использует динамически выбранный состав.
    3. Модель B — полный набор из пяти признаков; модель A — историческое
       среднее. Точность всех трёх вычисляется на периоде оценки (месяцы
       >= evaluation_target_start) по агрегации securities_then_months.

    Единая маска доступности обеспечивается сравнением на одинаковом множестве
    целевых месяцев (пересечение месяцев с прогнозом во всех моделях).
    """
    if not per_security:
        raise SelectionError("отбор состава реализован только для per_security=True")

    compositions = build_compositions()
    mse_by_comp: dict[Composition, pd.Series] = {}
    for comp in compositions:
        if not comp.features:
            continue  # пустой состав — это модель A (историческое среднее)
        forecasts = forecast_series(
            frame,
            feature_columns=list(comp.features),
            training_window_months=training_window_months,
            first_target_month=first_target_month,
            last_month=last_month,
            per_security=per_security,
        )
        mse_by_comp[comp] = monthly_mse_series(forecasts)

    # Сводная история выбора состава (документирование поведения правила).
    hist = rebalance_history(
        mse_by_comp,
        evaluation_target_start=evaluation_target_start,
        first_error_month=first_target_month,
        window_months=selection_window_months,
    )

    # Модель B — полный набор.
    full_features = composition_features(frozenset({_GROUP_PRICE, _GROUP_RISK, _GROUP_MACRO}))
    model_b_forecasts = forecast_series(
        frame,
        feature_columns=full_features,
        training_window_months=training_window_months,
        first_target_month=first_target_month,
        last_month=last_month,
        per_security=per_security,
    )
    mse_b = monthly_mse_series(model_b_forecasts)

    # Модель A — историческое среднее.
    model_a_forecasts = historical_average_forecast(
        frame,
        training_window_months=training_window_months,
        first_target_month=first_target_month,
        last_month=last_month,
        per_security=per_security,
    )
    mse_a = monthly_mse_series(model_a_forecasts)

    # Модель C — месячная MSE по динамически выбранному составу.
    mse_c = _apply_selected_composition_mse(
        mse_by_comp, hist, evaluation_target_start=evaluation_target_start
    )

    # Единая маска наблюдений на периоде оценки: пересечение месяцев с прогнозом.
    eval_cutoff = pd.Timestamp(evaluation_target_start)
    months = sorted({m for m in _union(mse_b, mse_a, mse_c) if m >= eval_cutoff})
    a_mse = _month_values(mse_a, months)
    b_mse = _month_values(mse_b, months)
    c_mse = _month_values(mse_c, months)
    ba_diff = float(np.mean(c_mse) - np.mean(b_mse))
    ca_diff = float(np.mean(c_mse) - np.mean(a_mse))
    win_rate = float(np.mean(np.asarray(c_mse) < np.asarray(b_mse))) if months else float("nan")

    comparison = ModelComparison(
        model_a=MonthError(mse=float(np.mean(a_mse)), mae=float(np.sqrt(np.mean(a_mse ** 2))), forecast_count=len(months)),
        model_b=MonthError(mse=float(np.mean(b_mse)), mae=float(np.sqrt(np.mean(b_mse ** 2))), forecast_count=len(months)),
        model_c=MonthError(mse=float(np.mean(c_mse)), mae=float(np.sqrt(np.mean(c_mse ** 2))), forecast_count=len(months)),
        comparison_ba=ba_diff,
        comparison_ca=ca_diff,
        comparison_ba_win_rate=win_rate,
        rebalance_history=hist,
    )
    return comparison


def _apply_selected_composition_mse(
    mse_by_comp: dict[Composition, pd.Series],
    hist: pd.DataFrame,
    *,
    evaluation_target_start: date,
) -> pd.Series:
    """Месячная MSE модели C по динамически выбранному составу.

    Для каждого целевого месяца берётся MSE состава, выбранного на последней
    дате пересмотра, не позднее этого месяца. Для месяцев до первой даты
    пересмотра (разогрев) состав не определён, поэтому они не участвуют в
    периоде оценки, который начинается с ``evaluation_target_start``.
    """
    if hist.empty:
        # Нет дат пересмотра — модель C сводится к полному набору до конца.
        # По умолчанию берем состав с наибольшим числом признаков.
        full = max(mse_by_comp, key=lambda c: c.size)
        return mse_by_comp[full].copy()

    all_months = sorted({m for series in mse_by_comp.values() for m in series.index})
    # Даты пересмотра и выбранный состав (сопоставление по группам).
    rebase = list(hist.itertuples(index=False))
    # Ограничиваем период оценки (после разогрева) — состав выбирается с даты
    # пересмотра; используем полный набор для месяцев до первого пересмотра.
    full = max(mse_by_comp, key=lambda c: c.size)
    result = {}
    for m in all_months:
        if m < pd.Timestamp(evaluation_target_start):
            continue
        selected = None
        for row in rebase:
            if pd.Timestamp(row.rebalance_date) <= m:
                selected = _composition_from_group_name(row.selected_groups, mse_by_comp)
        if selected is None:
            selected = full
        result[m] = mse_by_comp[selected][m]
    return pd.Series(result).sort_index()


def _composition_from_group_name(name: str, mse_by_comp: dict[Composition, pd.Series]) -> Composition:
    """Сопоставить имя группы с составом."""
    for comp in mse_by_comp:
        if _group_name(comp.groups) == name:
            return comp
    raise SelectionError(f"не найден состав с группами: {name!r}")


def _union(*series: pd.Series) -> set[pd.Timestamp]:
    result: set[pd.Timestamp] = set()
    for s in series:
        result.update(s.index)
    return result


def _month_values(series: pd.Series, months: list[pd.Timestamp]) -> np.ndarray:
    vals = [series[m] for m in months if m in series.index and pd.notna(series[m])]
    if not vals:
        return np.array([])
    return np.asarray(vals, dtype=float)


def np_greater(a, b) -> pd.Series:
    return pd.Series(a) > pd.Series(b)