"""Общие фикстуры для тестов этапа 8 (детерминизм, утечка)."""

from datetime import date

import pytest

from equity_factor_study.checks.synthetic import (
    build_synthetic_daily,
    build_synthetic_key_rate,
)

SYNTHETIC_SECIDS = ["SBER", "GAZP", "LKOH"]
SYNTHETIC_FROM = date(2014, 1, 1)
SYNTHETIC_TO = date(2014, 6, 1)


@pytest.fixture
def synthetic_daily():
    return build_synthetic_daily()


@pytest.fixture
def synthetic_key_rate():
    return build_synthetic_key_rate()


@pytest.fixture
def synthetic_secids():
    return SYNTHETIC_SECIDS


@pytest.fixture
def synthetic_range():
    return SYNTHETIC_FROM, SYNTHETIC_TO