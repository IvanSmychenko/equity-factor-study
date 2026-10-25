"""Тесты этапа 1: конфигурация, CLI и документация каркаса."""

from pathlib import Path

import pytest

from equity_factor_study.config import ConfigError, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


import re

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _write_and_load(tmp_path: Path, data: dict):
    """Записать модифицированную конфигурацию в TOML-файл и загрузить её.

    Даты при этом проходят через tomllib и становятся нативными ``datetime.date``,
    как в реальном сценарии использования.
    """
    path = tmp_path / "study.toml"
    path.write_text(_toml_dump(data), encoding="utf-8")
    return load_config(path)


def _toml_dump(data: dict) -> str:
    """Сериализовать небольшой словарь в TOML-строку (без зависимостей).

    Даты, заданные строками в формате YYYY-MM-DD, записываются как нативные
    TOML-даты (без кавычек) — как это делает реальный конфиг.
    """
    lines: list[str] = []
    for section, values in data.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            if value is True:
                lines.append(f"{key} = true")
            elif value is False:
                lines.append(f"{key} = false")
            elif isinstance(value, str) and _ISO_DATE.match(value):
                lines.append(f"{key} = {value}")
            elif isinstance(value, str):
                lines.append(f"{key} = {value!r}")
            elif isinstance(value, (list, tuple)):
                items = ", ".join(str(v) for v in value)
                lines.append(f"{key} = [{items}]")
            else:
                lines.append(f"{key} = {value}")
        lines.append("")
    return "\n".join(lines)


@pytest.fixture
def valid_config_data() -> dict:
    return {
        "protocol": {"version": "0.1"},
        "period": {
            "download_start": "2013-07-01",
            "observation_start": "2014-01-01",
            "observation_end": "2026-09-30",
            "development_target_end": "2020-12-31",
            "evaluation_target_start": "2021-01-01",
        },
        "universe": {
            "target_size": 10,
            "selection_board": "TQBR",
            "security_type": "common_share",
            "turnover_year": 2014,
            "selection_rule": "turnover_2014_desc",
            "require_survive_to_end": False,
            "allow_fewer": True,
        },
        "features": {
            "return_1m": True,
            "momentum_prev_3m": 3,
            "volatility_3m": 3,
            "annualize_volatility": False,
            "key_rate_pct": True,
            "key_rate_change_pp": True,
        },
        "month_end": {"max_trading_days_from_month_end": 5},
        "model": {
            "type": "ols",
            "per_security": True,
            "training_window_months": 48,
            "refit_frequency": "monthly",
            "window_sensitivity": [36, 60],
        },
        "selection": {
            "rebalance_frequency": "quarterly",
            "selection_window_months": 12,
            "selection_window_sensitivity": [18],
            "tie_break": "fewest_features",
            "level": "pooled_across_securities",
            "aggregation": "securities_then_months",
            "warmup_from": "development",
        },
        "evaluation": {
            "primary_metric": "mse",
            "secondary_metric": "mae",
            "aggregation": "securities_then_months",
        },
        "bootstrap": {
            "method": "paired_block",
            "block_months": 6,
            "block_sensitivity": [3, 12],
            "replicates": 5000,
            "confidence": 0.95,
            "seed": 20140101,
        },
    }


@pytest.fixture
def config_path(tmp_path, valid_config_data) -> Path:
    path = tmp_path / "study.toml"
    path.write_text(_toml_dump(valid_config_data), encoding="utf-8")
    return path


# --- успешная загрузка --------------------------------------------------------


@pytest.fixture
def real_config_path() -> Path:
    return PROJECT_ROOT / "configs" / "study.toml"


def test_project_config_loads_successfully(real_config_path):
    """Конфигурация проекта из configs/study.toml загружается без ошибок."""
    config = load_config(real_config_path)
    assert config.protocol.version == "0.1"
    assert config.first_observation_month.isoformat() == "2014-05-01"
    assert config.first_target_month.isoformat() == "2018-06-01"
    assert config.period_length_months == 153


def test_project_config_control_dates(real_config_path):
    """Контрольные даты пункта Б.6 воспроизводятся на данных по умолчанию."""
    config = load_config(real_config_path)
    assert config.first_observation_month.isoformat() == "2014-05-01"
    assert config.first_target_month.isoformat() == "2018-06-01"


def test_config_is_immutable(config_path):
    """Объект конфигурации неизменяем (frozen dataclasses)."""
    config = load_config(config_path)
    with pytest.raises(AttributeError):
        config.period.observation_start = None  # type: ignore[misc]


# --- отсутствие файла ---------------------------------------------------------


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="не найден"):
        load_config(tmp_path / "missing.toml")


def test_wrong_extension_raises(tmp_path):
    path = tmp_path / "study.txt"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="расширением .toml"):
        load_config(path)


def test_invalid_toml_raises(tmp_path):
    path = tmp_path / "study.toml"
    path.write_text("protocol = [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="Некорректный TOML"):
        load_config(path)


# --- неверный тип значения ----------------------------------------------------


@pytest.mark.parametrize(
    ("section", "key", "bad_value", "expected"),
    [
        ("features", "return_1m", "yes", "bool"),
        ("features", "momentum_prev_3m", "three", "int"),
        ("period", "observation_start", "2014/01/01", "date"),
        ("month_end", "max_trading_days_from_month_end", "many", "int"),
        ("bootstrap", "seed", "abc", "int"),
    ],
)
def test_wrong_type_raises(tmp_path, valid_config_data, section, key, bad_value, expected):
    data = {
        sect: dict(vals) for sect, vals in valid_config_data.items()
    }
    data[section][key] = bad_value

    with pytest.raises(ConfigError, match=expected):
        _write_and_load(tmp_path, data)


def test_zero_window_rejected(tmp_path, valid_config_data):
    """Окно нулевой длины отвергается (приложение Б, В3)."""
    data = {sect: dict(vals) for sect, vals in valid_config_data.items()}
    data["features"]["momentum_prev_3m"] = 0

    with pytest.raises(ConfigError, match="положительным целым"):
        _write_and_load(tmp_path, data)


# --- неизвестный ключ ---------------------------------------------------------


def test_unknown_key_in_section_raises(tmp_path, valid_config_data):
    data = {sect: dict(vals) for sect, vals in valid_config_data.items()}
    data["features"]["extra_key"] = True

    with pytest.raises(ConfigError, match="Неизвестные ключи в секции 'features'"):
        _write_and_load(tmp_path, data)


def test_unknown_top_level_section_raises(tmp_path, valid_config_data):
    data = {sect: dict(vals) for sect, vals in valid_config_data.items()}
    data["unknown_section"] = {}

    with pytest.raises(ConfigError, match="Неизвестные секции"):
        _write_and_load(tmp_path, data)


# --- взаимная согласованность параметров --------------------------------------


def test_download_lag_less_than_six_months_rejected(tmp_path, valid_config_data):
    data = {sect: dict(vals) for sect, vals in valid_config_data.items()}
    data["period"]["download_start"] = "2013-11-01"  # лаг 2 мес.

    with pytest.raises(ConfigError, match="download_start"):
        _write_and_load(tmp_path, data)


def test_development_end_after_evaluation_start_rejected(tmp_path, valid_config_data):
    data = {sect: dict(vals) for sect, vals in valid_config_data.items()}
    data["period"]["development_target_end"] = "2021-06-01"

    with pytest.raises(ConfigError, match="development_target_end"):
        _write_and_load(tmp_path, data)


def test_selection_window_exceeds_evaluation_rejected(tmp_path, valid_config_data):
    data = {sect: dict(vals) for sect, vals in valid_config_data.items()}
    data["selection"]["selection_window_months"] = 500  # больше 69 мес. оценки

    with pytest.raises(ConfigError, match="selection_window_months"):
        _write_and_load(tmp_path, data)


def test_seed_not_positive_rejected(tmp_path, valid_config_data):
    data = {sect: dict(vals) for sect, vals in valid_config_data.items()}
    data["bootstrap"]["seed"] = -5

    with pytest.raises(ConfigError, match="seed"):
        _write_and_load(tmp_path, data)


def test_window_fits_training_rejected(tmp_path, valid_config_data):
    """Окно обучения, которое нечем заполнить, отвергается (Б.5).

    От s0 (май 2014) до конца наблюдения доступно 148 завершённых пар, поэтому
    окно 200 мес. нечем заполнить.
    """
    data = {sect: dict(vals) for sect, vals in valid_config_data.items()}
    data["model"]["training_window_months"] = 200

    with pytest.raises(ConfigError, match="training_window_months"):
        _write_and_load(tmp_path, data)


def test_sensitivity_matches_main_value_rejected(tmp_path, valid_config_data):
    data = {sect: dict(vals) for sect, vals in valid_config_data.items()}
    data["model"]["window_sensitivity"] = [48, 36]

    with pytest.raises(ConfigError, match="не должен совпадать"):
        _write_and_load(tmp_path, data)