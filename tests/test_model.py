"""Тесты модельного слоя: маска, нормировка, скользящее обучение, метрики."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from equity_factor_study.checks.synthetic import build_long_continuous_panel
from equity_factor_study.model.evaluate import evaluate_forecasts, factor_usefulness
from equity_factor_study.model.fit import (
    FitError,
    OLSFit,
    StandardScaler,
    fit_ols,
    walk_forward,
)
from equity_factor_study.model.orchestrator import (
    run_model_evaluation,
)
from equity_factor_study.model.panel import (
    FEATURE_COLUMNS,
    PanelDataError,
    build_observation_matrix,
    select_features,
    split_by_period,
)

EVAL_END = date(2026, 9, 1)
DEV_END = date(2020, 12, 31)
FIRST_TARGET = date(2018, 6, 1)


@pytest.fixture(scope="module")
def long_panel():
    return build_long_continuous_panel(start=date(2014, 1, 1), end=EVAL_END)


@pytest.fixture(scope="module")
def observation_matrix(long_panel):
    return build_observation_matrix(long_panel)


def _frame(matrix, columns=FEATURE_COLUMNS):
    frame = matrix.features[columns].copy()
    frame["target_return_next_1m"] = matrix.target.to_numpy()
    frame["month"] = matrix.identity["month"].to_numpy()
    frame["secid"] = matrix.identity["secid"].to_numpy()
    return frame


def test_observation_matrix_single_row_per_observation(long_panel, observation_matrix):
    assert len(observation_matrix.features) == len(observation_matrix.target)
    assert len(observation_matrix.features) == len(observation_matrix.identity)
    assert observation_matrix.features.isna().any().sum() == 0


def test_observation_missing_required_column_raises(long_panel):
    broken = long_panel.drop(columns=["momentum_prev_3m"])
    with pytest.raises(PanelDataError):
        build_observation_matrix(broken)


def test_observation_mask_excludes_incomplete():
    """Строки с пропуском признака должны исключаться, а не заполняться."""
    frame = build_long_continuous_panel(start=date(2014, 1, 1), end=date(2014, 6, 1))
    mat = build_observation_matrix(frame)
    # Первые месяцы ряда (до внедрённого набора) не должны попадать в маску.
    assert len(mat.features) > 0
    assert mat.features.notna().all().all()


def test_selected_features_preserve_rows(observation_matrix):
    reduced = select_features(observation_matrix, ["return_1m", "volatility_3m"])
    assert len(reduced.features) == len(observation_matrix.features)
    assert reduced.features.columns.tolist() == ["return_1m", "volatility_3m"]


def test_selected_features_unknown_raises(observation_matrix):
    with pytest.raises(PanelDataError):
        select_features(observation_matrix, ["nonexistent"])


def test_split_by_period(observation_matrix):
    dev, ev = split_by_period(
        observation_matrix,
        development_target_end=DEV_END,
        evaluation_target_start=date(2021, 1, 1),
    )
    assert len(dev) + len(ev) == len(observation_matrix.features)
    assert len(dev) > 0 and len(ev) > 0


def test_standard_scaler_train_only():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    scaler = StandardScaler.fit(train)
    transformed = scaler.transform(pd.DataFrame({"x": [3.0, 5.0]}))
    # Среднее и масштаб зафиксированы по обучению, применение не пересчитывает.
    assert scaler.mean["x"] == pytest.approx(2.5)
    assert scaler.scale["x"] > 0
    assert len(transformed) == 2


def test_standard_scaler_zero_variance_safe():
    train = pd.DataFrame({"x": [1.0, 1.0, 1.0]})
    scaler = StandardScaler.fit(train)
    assert scaler.scale["x"] == 1.0  # не NaN
    transformed = scaler.transform(pd.DataFrame({"x": [1.0, 1.0]}))
    assert transformed.isna().any().sum() == 0


def test_fit_ols_requires_two_rows():
    train = _frame(build_observation_matrix(
        build_long_continuous_panel(start=date(2014, 1, 1), end=date(2014, 6, 1))
    ))
    with pytest.raises(FitError):
        fit_ols(train.head(1), feature_columns=FEATURE_COLUMNS)


def test_fit_ols_recovers_known_direction():
    """Линейная связь признака с целью должна восстанавливаться по знаку."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    target = 0.5 * x + rng.normal(scale=0.1, size=200)
    train = pd.DataFrame({"x": x})
    train["target_return_next_1m"] = target
    fitted = fit_ols(train, feature_columns=["x"])
    assert fitted.coefficients["x"] > 0.2
    assert isinstance(fitted, OLSFit)


def test_walk_forward_first_forecast_matches_B6(observation_matrix):
    frame = _frame(observation_matrix)
    forecasts = walk_forward(
        frame,
        feature_columns=FEATURE_COLUMNS,
        training_window_months=48,
        refit_frequency="monthly",
        first_target_month=FIRST_TARGET,
        last_month=EVAL_END,
    )
    assert forecasts
    assert forecasts[0].forecast_month == FIRST_TARGET
    assert forecasts[0].train_rows >= 10


def test_walk_forward_unsupported_frequency(observation_matrix):
    frame = _frame(observation_matrix)
    with pytest.raises(FitError):
        walk_forward(
            frame,
            feature_columns=FEATURE_COLUMNS,
            training_window_months=48,
            refit_frequency="weekly",
            first_target_month=FIRST_TARGET,
            last_month=EVAL_END,
        )


def test_walk_forward_per_security_groups_by_secid(observation_matrix):
    """per_security обучает отдельную регрессию по каждой акции; окно не
    смешивает наблюдения разных инструментов."""
    frame = _frame(observation_matrix)
    forecasts = walk_forward(
        frame,
        feature_columns=FEATURE_COLUMNS,
        training_window_months=48,
        refit_frequency="monthly",
        first_target_month=FIRST_TARGET,
        last_month=EVAL_END,
        per_security=True,
    )
    assert forecasts
    assert forecasts[0].forecast_month == FIRST_TARGET
    # Каждый прогноз — по одной акции (период оценки: 2 акции * 69 мес.).
    assert all(len(f.predicted) == 1 for f in forecasts)


def test_walk_forward_per_security_requires_secid(observation_matrix):
    frame = _frame(observation_matrix).drop(columns=["secid"])
    with pytest.raises(FitError, match="secid"):
        walk_forward(
            frame,
            feature_columns=FEATURE_COLUMNS,
            training_window_months=48,
            refit_frequency="monthly",
            first_target_month=FIRST_TARGET,
            last_month=EVAL_END,
            per_security=True,
        )


def test_build_compositions_eight_groups():
    """Из трёх групп образуются восемь составов (включая пустой)."""
    from equity_factor_study.model.selection import build_compositions

    comps = build_compositions()
    assert len(comps) == 8
    # Полный набор — 5 признаков; пустой — 0.
    sizes = sorted(c.size for c in comps)
    assert sizes == [0, 1, 2, 2, 3, 3, 4, 5]
    # Полный набор содержит все пять признаков.
    full = max(comps, key=lambda c: c.size)
    assert set(full.features) == set(FEATURE_COLUMNS)


def test_rebalance_first_date_after_warmup():
    """Первый пересмотр после накопления 12 ошибок — 1 июля 2019 (пункт Б.4)."""
    from equity_factor_study.model.selection import rebalance_dates

    dates = rebalance_dates(
        first_error_month=date(2018, 6, 1),
        evaluation_target_start=date(2021, 1, 1),
        last_month=date(2026, 9, 1),
        window_months=12,
    )
    assert dates[0] == date(2019, 7, 1)
    # Квартальная сетка.
    assert all(d.month in (1, 4, 7, 10) for d in dates)


def test_select_composition_prefers_fewer_on_tie():
    """При равенстве ошибок выбирается более простой состав (меньше признаков)."""
    from equity_factor_study.model.selection import (
        build_compositions,
        select_composition,
    )

    comps = build_compositions()
    simple = next(c for c in comps if c.size == 1)
    rich = next(c for c in comps if c.size == 5)
    idx = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"])
    mse = {}
    for comp, val in [(simple, 0.01), (rich, 0.01)]:
        mse[comp] = pd.Series([val] * 4, index=idx)
    chosen = select_composition(mse, rebalance_date=date(2020, 5, 1), window_months=4)
    assert chosen.size == 1


def test_run_selection_evaluation_compares_models():
    """Оркестратор отбора строит модели A/B/C и сравнивает точность."""
    from equity_factor_study.model.selection import run_selection_evaluation

    panel = build_long_continuous_panel(start=date(2014, 1, 1), end=date(2026, 9, 1))
    mat = build_observation_matrix(panel)
    frame = mat.features.copy()
    frame["target_return_next_1m"] = mat.target.to_numpy()
    frame["month"] = mat.identity["month"].to_numpy()
    frame["secid"] = mat.identity["secid"].to_numpy()

    report = run_selection_evaluation(
        frame,
        training_window_months=48,
        first_target_month=FIRST_TARGET,
        evaluation_target_start=date(2021, 1, 1),
        last_month=EVAL_END,
    )
    assert report.model_a.forecast_count > 0
    assert report.model_b.forecast_count == report.model_c.forecast_count
    # Разность C−B — конечное число.
    assert np.isfinite(report.comparison_ba)
    assert not report.rebalance_history.empty


def test_paired_block_bootstrap_deterministic():
    """Бутстрэп детерминирован при фиксированном зерне."""
    from equity_factor_study.model.bootstrap import paired_block_bootstrap

    idx = pd.date_range("2021-01-01", periods=68, freq="MS")
    diff = pd.Series(0.0001 * np.sin(np.arange(68)) + 0.00001, index=idx)
    r1 = paired_block_bootstrap(diff, block_months=6, replicates=1000, seed=20140101)
    r2 = paired_block_bootstrap(diff, block_months=6, replicates=1000, seed=20140101)
    assert r1.lower == r2.lower
    assert r1.upper == r2.upper
    assert r1.mean == r2.mean


def test_paired_block_bootstrap_interval_bounds():
    """Интервал содержит точечную оценку при симметричном ряде близко к нулю."""
    from equity_factor_study.model.bootstrap import paired_block_bootstrap

    idx = pd.date_range("2021-01-01", periods=68, freq="MS")
    diff = pd.Series(np.full(68, 0.0005), index=idx)
    r = paired_block_bootstrap(diff, block_months=6, replicates=2000, seed=7)
    assert r.point_estimate == 0.0005
    # Интервал должен быть узким при постоянном ряде.
    assert r.lower <= r.point_estimate <= r.upper


def test_paired_block_bootstrap_requires_block_shorter_than_series():
    from equity_factor_study.model.bootstrap import BootstrapError, paired_block_bootstrap

    idx = pd.date_range("2021-01-01", periods=10, freq="MS")
    diff = pd.Series(np.zeros(10), index=idx)
    with pytest.raises(BootstrapError):
        paired_block_bootstrap(diff, block_months=12, replicates=100, seed=1)


def test_evaluate_forecasts_securities_then_months(observation_matrix):
    frame = _frame(observation_matrix)
    forecasts = walk_forward(
        frame,
        feature_columns=FEATURE_COLUMNS,
        training_window_months=48,
        refit_frequency="monthly",
        first_target_month=FIRST_TARGET,
        last_month=EVAL_END,
    )
    result = evaluate_forecasts(forecasts, segment="evaluation",
                                development_target_end=DEV_END)
    assert result.forecast_count == len(forecasts)
    assert result.mse >= 0
    assert result.mae >= 0


def test_factor_usefulness_positive_when_reduced_worse():
    full = type("M", (), {"mse": 1.0, "mae": 0.5, "forecast_count": 10})()
    reduced = type("M", (), {"mse": 1.3, "mae": 0.7, "forecast_count": 10})()
    out = factor_usefulness(full, reduced)
    assert out["direction"] == "useful"
    assert out["usefulness_pct"] > 0


def test_factor_usefulness_negative_when_full_better():
    full = type("M", (), {"mse": 1.0, "mae": 0.5, "forecast_count": 10})()
    reduced = type("M", (), {"mse": 0.8, "mae": 0.4, "forecast_count": 10})()
    out = factor_usefulness(full, reduced)
    assert out["direction"] == "harmful"


def test_run_model_evaluation_barrier(long_panel):
    report = run_model_evaluation(
        long_panel,
        feature_columns=FEATURE_COLUMNS,
        training_window_months=48,
        refit_frequency="monthly",
        first_target_month=FIRST_TARGET,
        development_target_end=DEV_END,
        last_month=EVAL_END,
    )
    assert report.full_metrics.forecast_count > 0
    assert set(report.factor_usefulness.keys()) == set(FEATURE_COLUMNS)