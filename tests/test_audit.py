"""Тесты этапа 6: диагностика качества и корпоративные события (audit)."""

from datetime import date

import pandas as pd
import pytest

from equity_factor_study.pipelines.audit import (
    EVENT_COLUMNS,
    EVENT_SPLIT,
    AuditError,
    build_events_index,
    load_corporate_events,
    run_audit,
    write_events,
    write_findings,
    write_summary,
)


def _prices(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["secid", "trade_date", "close", "value"])


def test_load_corporate_events_missing_returns_empty(tmp_path):
    events = load_corporate_events(tmp_path / "corporate_actions.csv")
    assert events.empty
    assert list(events.columns) == EVENT_COLUMNS


def test_load_corporate_events_parses_dates(tmp_path):
    path = tmp_path / "corporate_actions.csv"
    path.write_text("secid,date,event_type\nA,2014-06-01,split\n", encoding="utf-8")
    events = load_corporate_events(path)
    assert events.iloc[0]["date"] == date(2014, 6, 1)
    assert events.iloc[0]["event_type"] == EVENT_SPLIT


def test_load_corporate_events_missing_column_raises(tmp_path):
    path = tmp_path / "corporate_actions.csv"
    path.write_text("secid,date\nA,2014-06-01\n", encoding="utf-8")
    with pytest.raises(AuditError, match="event_type"):
        load_corporate_events(path)


def test_build_events_index():
    events = pd.DataFrame(
        {
            "secid": ["A", "B"],
            "date": [date(2014, 6, 1), date(2015, 3, 15)],
            "event_type": [EVENT_SPLIT, EVENT_SPLIT],
        }
    )
    index = build_events_index(events)
    assert ("A", "2014-06-01") in index
    assert ("B", "2015-03-15") in index


def test_run_audit_collects_all_checks():
    prices = _prices(
        [
            {"secid": "A", "trade_date": date(2014, 1, 6), "close": 10.0, "value": 100.0},
            {"secid": "A", "trade_date": date(2014, 1, 7), "close": 0.0, "value": 100.0},  # неположительная
            {"secid": "A", "trade_date": date(2014, 1, 8), "close": 90.0, "value": 100.0},  # скачок 800%
        ]
    )
    events = pd.DataFrame(
        {"secid": ["B"], "date": [date(2014, 1, 8)], "event_type": [EVENT_SPLIT]}
    )
    result = run_audit(prices, events=events)
    assert result.findings.iloc[0]["secid"] == "A"
    assert len(result.events) == 1
    assert not result.summary.empty
    assert result.summary["count"].sum() == len(result.findings)


def test_run_audit_large_change_excluded_by_event():
    prices = _prices(
        [
            {"secid": "A", "trade_date": date(2014, 1, 6), "close": 10.0, "value": 100.0},
            {"secid": "A", "trade_date": date(2014, 1, 7), "close": 100.0, "value": 100.0},
        ]
    )
    events = pd.DataFrame(
        {"secid": ["A"], "date": [date(2014, 1, 7)], "event_type": [EVENT_SPLIT]}
    )
    result = run_audit(prices, events=events)
    # Скачок на дате события не является дефектом.
    assert result.findings.empty


def test_run_audit_clean_data_no_findings():
    prices = _prices(
        [
            {"secid": "A", "trade_date": date(2014, 1, 6), "close": 10.0, "value": 100.0},
            {"secid": "A", "trade_date": date(2014, 1, 7), "close": 10.2, "value": 100.0},
        ]
    )
    result = run_audit(prices)
    assert result.findings.empty
    assert result.summary.empty


def test_write_reports(tmp_path):
    findings = pd.DataFrame({"secid": ["A"], "date": ["2014-01-07"], "type": ["x"], "value": [1.0]})
    events = pd.DataFrame({"secid": ["A"], "date": [date(2014, 1, 7)], "event_type": [EVENT_SPLIT]})
    summary = pd.DataFrame({"type": ["x"], "count": [1]})

    f = write_findings(findings, tmp_path / "findings.csv")
    e = write_events(events, tmp_path / "corporate_events.csv")
    s = write_summary(summary, tmp_path / "quality_summary.csv")

    assert f.exists() and e.exists() and s.exists()
    assert "2014-01-07" in e.read_text(encoding="utf-8")