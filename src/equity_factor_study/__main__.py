"""Точка входа ``python -m equity_factor_study``."""

from __future__ import annotations

import sys

from equity_factor_study.cli import main

if __name__ == "__main__":
    sys.exit(main())