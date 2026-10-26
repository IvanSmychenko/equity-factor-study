"""Тесты этапа 8, проверка самих проверок (А.6).

В код поочерёдно вносятся заведомые дефекты, и набор обязан обнаружить
каждый из них. Подмена построителя панели выполняется через monkeypatch
модуля, к которому обращается соответствующая проверка, чтобы дефект прошёл
через её логику.
"""

from datetime import date

import pytest

import equity_factor_study.pipelines.monthly as monthly_mod
from equity_factor_study.checks import defects as df
from equity_factor_study.checks import leakage as lk

# Для обнаружения дефектов сдвига нужен диапазон, при котором искажение после D
# затрагивает наблюдаемый месяц (короткие диапазоны маскируют дефект на границе).
DEFECT_FROM = date(2014, 1, 1)
DEFECT_TO = date(2015, 12, 1)


@pytest.fixture
def defect_range():
    return DEFECT_FROM, DEFECT_TO


def test_defect_return_next_month_detected_by_a4(monkeypatch, synthetic_daily,
                                                 synthetic_key_rate,
                                                 synthetic_secids, defect_range):
    from_month, to_month = defect_range
    assert df.DEFECTS[0][1] is df.defect_return_uses_next_month
    monkeypatch.setattr(lk, "build_monthly_panel", df.defect_return_uses_next_month)
    result = lk.a4_main_leak_check(
        synthetic_daily, synthetic_secids, from_month, to_month,
        key_rate=synthetic_key_rate,
    )
    assert result["passed"] is False, "дефект сдвига доходности не обнаружен"


def test_defect_momentum_window_3_detected_by_tl2(monkeypatch):
    monkeypatch.setattr(monthly_mod, "build_monthly_panel", df.defect_momentum_window_3)
    result = lk.tl_2_formula_exact_match(None, None)
    assert result["passed"] is False, "дефект окна моментума не обнаружен"


def test_defect_volatility_full_period_detected(monkeypatch, synthetic_daily,
                                                synthetic_secids, defect_range):
    """Дефект волатильности по всему периоду обнаруживается сверкой с эталоном."""
    from_month, to_month = defect_range
    expected = lk._panel(
        daily=synthetic_daily, secids=synthetic_secids, from_month=from_month,
        to_month=to_month, max_trading_days=5, momentum_window=3, volatility_window=3,
        key_rate=None,
    )
    defective = df.defect_volatility_full_period(
        synthetic_daily, secids=synthetic_secids, from_month=from_month,
        to_month=to_month, max_trading_days=5, momentum_window=3, volatility_window=3,
        key_rate=None,
    )
    merged = defective.merge(
        expected, on=["secid", "month"], suffixes=("_d", "_e")
    )
    differ = lk._mismatches(merged["volatility_3m_d"], merged["volatility_3m_e"])
    assert differ.any(), "дефект волатильности по всему периоду не обнаружен"


def test_defect_key_rate_next_month_detected_by_tl4(monkeypatch, synthetic_daily,
                                                    synthetic_key_rate,
                                                    synthetic_secids, defect_range):
    from_month, to_month = defect_range
    monkeypatch.setattr(lk, "build_monthly_panel", df.defect_key_rate_next_month)
    result = lk.tl_4_key_rate_convention(
        synthetic_daily, synthetic_secids, from_month, to_month,
        key_rate=synthetic_key_rate,
    )
    assert result["passed"] is False, "дефект ставки следующего месяца не обнаружен"


def test_all_known_defects_declared():
    assert len(df.DEFECTS) == 4
    ids = {d[0] for d in df.DEFECTS}
    assert ids == {
        "D-RETURN-NEXT",
        "D-MOMENTUM-3",
        "D-VOLATILITY-FULL",
        "D-KEYRATE-NEXT",
    }
    for info in df.DEFECTS:
        assert info[0] in df.DETECTORS