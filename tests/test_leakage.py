"""Тесты этапа 8, класс проверок отсутствия утечки (А.4, TL-1..TL-9)."""


from equity_factor_study.checks import leakage as lk


def test_a4_main_leak_check(synthetic_daily, synthetic_key_rate,
                            synthetic_secids, synthetic_range):
    from_month, to_month = synthetic_range
    result = lk.a4_main_leak_check(
        synthetic_daily, synthetic_secids, from_month, to_month,
        key_rate=synthetic_key_rate,
    )
    assert result["passed"], result["detail"]


def test_tl_1_truncated_equals_full(synthetic_daily, synthetic_key_rate,
                                    synthetic_secids, synthetic_range):
    from_month, to_month = synthetic_range
    result = lk.tl_1_truncated_equals_full(
        synthetic_daily, synthetic_secids, from_month, to_month,
        key_rate=synthetic_key_rate,
    )
    assert result["passed"], result["detail"]


def test_tl_2_formula_exact_match():
    result = lk.tl_2_formula_exact_match(None, None)
    assert result["passed"], result["detail"]


def test_tl_3_availability_mask(synthetic_daily, synthetic_key_rate,
                                synthetic_secids, synthetic_range):
    from_month, to_month = synthetic_range
    result = lk.tl_3_availability_mask(
        synthetic_daily, synthetic_secids, from_month, to_month,
        key_rate=synthetic_key_rate,
    )
    assert result["passed"], result["detail"]


def test_tl_4_key_rate_convention(synthetic_daily, synthetic_key_rate,
                                  synthetic_secids, synthetic_range):
    from_month, to_month = synthetic_range
    result = lk.tl_4_key_rate_convention(
        synthetic_daily, synthetic_secids, from_month, to_month,
        key_rate=synthetic_key_rate,
    )
    assert result["passed"], result["detail"]


def test_tl_5_stale_price_flag(synthetic_daily, synthetic_key_rate,
                               synthetic_secids, synthetic_range):
    from_month, to_month = synthetic_range
    result = lk.tl_5_stale_price_flag(
        synthetic_daily, synthetic_secids, from_month, to_month,
        key_rate=synthetic_key_rate,
    )
    assert result["passed"], result["detail"]


def test_tl_6_split_flagged_not_patched(synthetic_daily, synthetic_key_rate,
                                        synthetic_secids, synthetic_range):
    from_month, to_month = synthetic_range
    result = lk.tl_6_split_flagged_not_patched(
        synthetic_daily, synthetic_secids, from_month, to_month,
        key_rate=synthetic_key_rate,
    )
    assert result["passed"], result["detail"]


def test_tl_7_http_failure_does_not_break(synthetic_daily, synthetic_key_rate,
                                          synthetic_secids, synthetic_range):
    from_month, to_month = synthetic_range
    result = lk.tl_7_http_failure_does_not_break(
        synthetic_daily, synthetic_secids, from_month, to_month,
        key_rate=synthetic_key_rate,
    )
    assert result["passed"], result["detail"]


def test_tl_8_early_features_marked(synthetic_daily, synthetic_key_rate,
                                    synthetic_secids, synthetic_range):
    from_month, to_month = synthetic_range
    result = lk.tl_8_early_features_marked(
        synthetic_daily, synthetic_secids, from_month, to_month,
        key_rate=synthetic_key_rate,
    )
    assert result["passed"], result["detail"]


def test_tl_9_control_dates():
    result = lk.tl_9_control_dates()
    assert result["passed"], result["detail"]
    assert result["rows_compared"] == 100