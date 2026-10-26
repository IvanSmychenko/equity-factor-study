"""Тесты этапа 8, класс проверок детерминизма (TD-1..TD-7, А.3)."""


from equity_factor_study.checks import determinism as d


def test_td_1_two_runs_identical(synthetic_daily, synthetic_key_rate,
                                 synthetic_secids, synthetic_range):
    from_month, to_month = synthetic_range
    result = d.td_1_two_runs_identical(
        synthetic_daily, synthetic_secids, from_month, to_month,
        key_rate=synthetic_key_rate,
    )
    assert result["passed"], result["detail"]


def test_td_2_row_order_does_not_matter(synthetic_daily, synthetic_key_rate,
                                        synthetic_secids, synthetic_range):
    from_month, to_month = synthetic_range
    result = d.td_2_row_order_does_not_matter(
        synthetic_daily, synthetic_secids, from_month, to_month,
        key_rate=synthetic_key_rate,
    )
    assert result["passed"], result["detail"]


def test_td_3_reverse_file_order(synthetic_daily, synthetic_key_rate,
                                 synthetic_secids, synthetic_range):
    from_month, to_month = synthetic_range
    result = d.td_3_reverse_file_order(
        synthetic_daily, synthetic_secids, from_month, to_month,
        key_rate=synthetic_key_rate,
    )
    assert result["passed"], result["detail"]


def test_td_4_timezone_locale_invariant(synthetic_daily, synthetic_key_rate,
                                        synthetic_secids, synthetic_range):
    from_month, to_month = synthetic_range
    result = d.td_4_timezone_locale_invariant(
        synthetic_daily, synthetic_secids, from_month, to_month,
        key_rate=synthetic_key_rate,
    )
    assert result["passed"], result["detail"]


def test_td_5_system_time_invariant(synthetic_daily, synthetic_key_rate,
                                    synthetic_secids, synthetic_range):
    from_month, to_month = synthetic_range
    result = d.td_5_system_time_invariant(
        synthetic_daily, synthetic_secids, from_month, to_month,
        key_rate=synthetic_key_rate,
    )
    assert result["passed"], result["detail"]


def test_td_6_no_overwrite_and_unique_run_dir(tmp_path):
    result = d.td_6_no_overwrite(tmp_path)
    assert result["passed"], result["detail"]


def test_td_7_source_hashes_stable(tmp_path):
    src = tmp_path / "daily_prices.csv"
    src.write_text("secid,close\nA,1.0\n")
    result = d.td_7_source_hashes_stable([str(src)])
    assert result["passed"], result["detail"]


def test_manifest_without_timestamps_strips_fields():
    manifest = {"protocol_version": "0.1", "timestamp": "2024-01-01T00:00:00Z",
                "nested": {"generated_at": "2024-01-01T00:00:00Z"}}
    scrubbed = d.manifest_without_timestamps(manifest)
    assert "timestamp" not in scrubbed
    assert "generated_at" not in scrubbed["nested"]