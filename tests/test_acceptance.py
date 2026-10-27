"""Тесты приёмки этапа 8 (А.7): отчёт перечисляет все проверки и контрольные даты."""


from equity_factor_study.checks.acceptance import run_all_checks, write_report


def test_run_all_checks_covers_all_sections():
    summary = run_all_checks()
    results = summary["results"]
    ids = {r["id"] for r in results}
    # Все проверки детерминизма и утечки должны присутствовать.
    expected = {
        "TD-1", "TD-2", "TD-3", "TD-4", "TD-5", "TD-6", "TD-7",
        "А.4", "TL-1", "TL-2", "TL-3", "TL-4", "TL-5", "TL-6", "TL-7",
        "TL-8", "TL-9", "А.6",
    }
    assert expected.issubset(ids)


def test_run_all_checks_acceptable():
    summary = run_all_checks()
    assert all(r["passed"] for r in summary["results"])


def test_control_dates_match_b6():
    summary = run_all_checks()
    dates = summary["control_dates"]
    assert dates["first_observation_month"] == "2014-05-01"
    assert dates["first_target_month"] == "2018-06-01"
    assert dates["development_predictions"] == 31
    assert dates["evaluation_predictions"] == 69
    assert dates["total_predictions"] == 100
    assert dates["first_window_pairs"] == 48


def test_write_report_creates_md_and_json(tmp_path):
    md = write_report(tmp_path)
    assert md.exists()
    assert (tmp_path / "acceptance_report.json").exists()
    content = md.read_text(encoding="utf-8")
    assert "Всего проверок" in content
    assert "[PASS]" in content


def test_check_required_inputs(tmp_path):
    """Проверка обязательных входов: недостающий вход фиксируется как расхождение."""
    from equity_factor_study.checks.acceptance import check_required_inputs

    present = tmp_path / "daily_prices.csv"
    present.write_text("x\n", encoding="utf-8")
    missing = tmp_path / "corporate_actions.csv"

    results = check_required_inputs(
        [("цены", present), ("события", missing)]
    )
    assert len(results) == 2
    by_desc = {r["description"]: r for r in results}
    assert by_desc["обязательный вход цены"]["passed"] is True
    assert by_desc["обязательный вход события"]["passed"] is False


def test_write_report_fails_on_missing_input(tmp_path):
    """Приёмка не принимается, если обязательный вход отсутствует (шаг A1)."""
    md = write_report(
        tmp_path,
        input_paths=[("события", tmp_path / "corporate_actions.csv")],
    )
    content = md.read_text(encoding="utf-8")
    assert "[FAIL]" in content
    assert "не принято" in content