"""Загрузка и валидация конфигурации исследования.

Модуль читает ``configs/study.toml``, проверяет типы, диапазоны и взаимную
согласованность параметров (приложение Б технического задания) и возвращает
неизменяемый объект конфигурации. Неизвестные ключи считаются ошибкой.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

# Дата, с которой доступен признак "доходность за месяц" по отношению к началу
# периода наблюдения: для первого месяца ряда доходность неопределима,
# поэтому смещение равно одному месяцу.
_RETURN_OFFSET_MONTHS = 1


class ConfigError(Exception):
    """Ошибка конфигурации с точным указанием проблемного параметра."""


@dataclass(frozen=True)
class ProtocolConfig:
    version: str


@dataclass(frozen=True)
class PeriodConfig:
    download_start: date
    observation_start: date
    observation_end: date
    development_target_end: date
    evaluation_target_start: date


@dataclass(frozen=True)
class UniverseConfig:
    target_size: int
    selection_board: str
    security_type: str
    turnover_year: int
    selection_rule: str
    require_survive_to_end: bool
    allow_fewer: bool


@dataclass(frozen=True)
class FeaturesConfig:
    return_1m: bool
    momentum_prev_3m: int
    volatility_3m: int
    annualize_volatility: bool
    key_rate_pct: bool
    key_rate_change_pp: bool


@dataclass(frozen=True)
class MonthEndConfig:
    max_trading_days_from_month_end: int


@dataclass(frozen=True)
class ModelConfig:
    type: str
    per_security: bool
    training_window_months: int
    refit_frequency: str
    window_sensitivity: tuple[int, ...]


@dataclass(frozen=True)
class SelectionConfig:
    rebalance_frequency: str
    selection_window_months: int
    selection_window_sensitivity: tuple[int, ...]
    tie_break: str
    level: str
    aggregation: str
    warmup_from: str


@dataclass(frozen=True)
class EvaluationConfig:
    primary_metric: str
    secondary_metric: str
    aggregation: str


@dataclass(frozen=True)
class BootstrapConfig:
    method: str
    block_months: int
    block_sensitivity: tuple[int, ...]
    replicates: int
    confidence: float
    seed: int


@dataclass(frozen=True)
class StudyConfig:
    protocol: ProtocolConfig
    period: PeriodConfig
    universe: UniverseConfig
    features: FeaturesConfig
    month_end: MonthEndConfig
    model: ModelConfig
    selection: SelectionConfig
    evaluation: EvaluationConfig
    bootstrap: BootstrapConfig

    # --- производные значения ------------------------------------------------

    @property
    def period_length_months(self) -> int:
        """Число календарных месяцев периода наблюдения (включая границы)."""
        return (
            (self.period.observation_end.year - self.period.observation_start.year) * 12
            + self.period.observation_end.month
            - self.period.observation_start.month
            + 1
        )

    @property
    def first_observation_month(self) -> date:
        """Начало месяца наблюдения s0 с полным набором из пяти признаков.

        Смещение определяется самым поздним признаком: моментум требует цены
        на ``window + 1`` месяц назад, волатильность — ``window - 1`` месячный
        интервал, доходность — один месяц. Итоговое смещение — максимум из них.
        """
        momentum_offset = self.features.momentum_prev_3m + 1
        volatility_offset = self.features.volatility_3m - 1
        return _shift_months(
            self.period.observation_start,
            max(_RETURN_OFFSET_MONTHS, momentum_offset, volatility_offset),
        )

    @property
    def first_target_month(self) -> date:
        """Первый целевой месяц прогноза: s0 + W + 1 (пункт Б.6)."""
        return _shift_months(
            self.period.observation_start,
            self._s0_offset_months() + self.model.training_window_months + 1,
        )

    @property
    def full_feature_set_size(self) -> int:
        """Число признаков полного набора (всегда пять)."""
        return 5

    def _s0_offset_months(self) -> int:
        momentum_offset = self.features.momentum_prev_3m + 1
        volatility_offset = self.features.volatility_3m - 1
        return max(_RETURN_OFFSET_MONTHS, momentum_offset, volatility_offset)


def _shift_months(value: date, months: int) -> date:
    """Сдвинуть дату на число календарных месяцев, сохраняя день (обычно 1-е)."""
    total = value.year * 12 + (value.month - 1) + months
    year, month_index = divmod(total, 12)
    return value.replace(year=year, month=month_index + 1, day=1)


# --- базовая валидация типов --------------------------------------------------


def _require_key(section: dict[str, Any], key: str, section_name: str) -> None:
    if key not in section:
        raise ConfigError(f"Отсутствует обязательный ключ '{section_name}.{key}'")


def _require_type(
    value: Any,
    expected: type | tuple[type, ...],
    path: str,
) -> None:
    if not isinstance(value, expected):
        expected_names = (
            ", ".join(t.__name__ for t in expected) if isinstance(expected, tuple) else expected.__name__
        )
        raise ConfigError(
            f"Параметр '{path}' имеет неверный тип: ожидался {expected_names}, "
            f"получено {type(value).__name__}"
        )


def _require_date(value: Any, path: str) -> date:
    _require_type(value, date, path)
    return value


def _require_bool(value: Any, path: str) -> bool:
    # tomllib отдаёт настоящие bool, поэтому достаточно strict-проверки.
    _require_type(value, bool, path)
    return value


def _require_positive_int(value: Any, path: str) -> int:
    _require_type(value, int, path)
    if value <= 0:
        raise ConfigError(f"Параметр '{path}' должен быть положительным целым, получено {value}")
    return value


def _require_int(value: Any, path: str) -> int:
    _require_type(value, int, path)
    return value


def _require_str(value: Any, path: str) -> str:
    _require_type(value, str, path)
    return value


def _require_sensitivity(
    value: Any,
    path: str,
    main_value: int,
) -> tuple[int, ...]:
    _require_type(value, list, path)
    if len(value) == 0:
        raise ConfigError(f"Параметр '{path}' не может быть пустым списком")
    result: list[int] = []
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        _require_type(item, int, item_path)
        if item <= 0:
            raise ConfigError(f"Параметр '{item_path}' должен быть положительным целым")
        if item == main_value:
            raise ConfigError(
                f"Параметр '{item_path}' не должен совпадать с основным значением "
                f"({main_value}): значения чувствительности отличаются от основного"
            )
        result.append(item)
    return tuple(result)


def _reject_unknown_keys(section: dict[str, Any], allowed: set[str], section_name: str) -> None:
    unknown = set(section) - allowed
    if unknown:
        keys = ", ".join(sorted(unknown))
        raise ConfigError(f"Неизвестные ключи в секции '{section_name}': {keys}")


# --- парсинг секций -----------------------------------------------------------


def _parse_protocol(raw: dict[str, Any]) -> ProtocolConfig:
    _reject_unknown_keys(raw, {"version"}, "protocol")
    return ProtocolConfig(
        version=_require_str(_required(raw, "version", "protocol"), "protocol.version"),
    )


def _parse_period(raw: dict[str, Any]) -> PeriodConfig:
    allowed = {
        "download_start",
        "observation_start",
        "observation_end",
        "development_target_end",
        "evaluation_target_start",
    }
    _reject_unknown_keys(raw, allowed, "period")
    return PeriodConfig(
        download_start=_require_date(_required(raw, "download_start", "period"), "period.download_start"),
        observation_start=_require_date(_required(raw, "observation_start", "period"), "period.observation_start"),
        observation_end=_require_date(_required(raw, "observation_end", "period"), "period.observation_end"),
        development_target_end=_require_date(
            _required(raw, "development_target_end", "period"), "period.development_target_end"
        ),
        evaluation_target_start=_require_date(
            _required(raw, "evaluation_target_start", "period"), "period.evaluation_target_start"
        ),
    )


def _parse_universe(raw: dict[str, Any]) -> UniverseConfig:
    allowed = {
        "target_size",
        "selection_board",
        "security_type",
        "turnover_year",
        "selection_rule",
        "require_survive_to_end",
        "allow_fewer",
    }
    _reject_unknown_keys(raw, allowed, "universe")
    return UniverseConfig(
        target_size=_require_positive_int(_required(raw, "target_size", "universe"), "universe.target_size"),
        selection_board=_require_str(_required(raw, "selection_board", "universe"), "universe.selection_board"),
        security_type=_require_str(_required(raw, "security_type", "universe"), "universe.security_type"),
        turnover_year=_require_int(_required(raw, "turnover_year", "universe"), "universe.turnover_year"),
        selection_rule=_require_str(_required(raw, "selection_rule", "universe"), "universe.selection_rule"),
        require_survive_to_end=_require_bool(
            _required(raw, "require_survive_to_end", "universe"), "universe.require_survive_to_end"
        ),
        allow_fewer=_require_bool(_required(raw, "allow_fewer", "universe"), "universe.allow_fewer"),
    )


def _parse_features(raw: dict[str, Any]) -> FeaturesConfig:
    allowed = {
        "return_1m",
        "momentum_prev_3m",
        "volatility_3m",
        "annualize_volatility",
        "key_rate_pct",
        "key_rate_change_pp",
    }
    _reject_unknown_keys(raw, allowed, "features")
    return FeaturesConfig(
        return_1m=_require_bool(_required(raw, "return_1m", "features"), "features.return_1m"),
        momentum_prev_3m=_require_positive_int(
            _required(raw, "momentum_prev_3m", "features"), "features.momentum_prev_3m"
        ),
        volatility_3m=_require_positive_int(_required(raw, "volatility_3m", "features"), "features.volatility_3m"),
        annualize_volatility=_require_bool(
            _required(raw, "annualize_volatility", "features"), "features.annualize_volatility"
        ),
        key_rate_pct=_require_bool(_required(raw, "key_rate_pct", "features"), "features.key_rate_pct"),
        key_rate_change_pp=_require_bool(
            _required(raw, "key_rate_change_pp", "features"), "features.key_rate_change_pp"
        ),
    )


def _parse_month_end(raw: dict[str, Any]) -> MonthEndConfig:
    _reject_unknown_keys(raw, {"max_trading_days_from_month_end"}, "month_end")
    return MonthEndConfig(
        max_trading_days_from_month_end=_require_positive_int(
            _required(raw, "max_trading_days_from_month_end", "month_end"),
            "month_end.max_trading_days_from_month_end",
        ),
    )


def _parse_model(raw: dict[str, Any]) -> ModelConfig:
    allowed = {"type", "per_security", "training_window_months", "refit_frequency", "window_sensitivity"}
    _reject_unknown_keys(raw, allowed, "model")
    training_window_months = _require_positive_int(
        _required(raw, "training_window_months", "model"), "model.training_window_months"
    )
    return ModelConfig(
        type=_require_str(_required(raw, "type", "model"), "model.type"),
        per_security=_require_bool(_required(raw, "per_security", "model"), "model.per_security"),
        training_window_months=training_window_months,
        refit_frequency=_require_str(_required(raw, "refit_frequency", "model"), "model.refit_frequency"),
        window_sensitivity=_require_sensitivity(
            _required(raw, "window_sensitivity", "model"),
            "model.window_sensitivity",
            training_window_months,
        ),
    )


def _parse_selection(raw: dict[str, Any]) -> SelectionConfig:
    allowed = {
        "rebalance_frequency",
        "selection_window_months",
        "selection_window_sensitivity",
        "tie_break",
        "level",
        "aggregation",
        "warmup_from",
    }
    _reject_unknown_keys(raw, allowed, "selection")
    selection_window_months = _require_positive_int(
        _required(raw, "selection_window_months", "selection"), "selection.selection_window_months"
    )
    return SelectionConfig(
        rebalance_frequency=_require_str(
            _required(raw, "rebalance_frequency", "selection"), "selection.rebalance_frequency"
        ),
        selection_window_months=selection_window_months,
        selection_window_sensitivity=_require_sensitivity(
            _required(raw, "selection_window_sensitivity", "selection"),
            "selection.selection_window_sensitivity",
            selection_window_months,
        ),
        tie_break=_require_str(_required(raw, "tie_break", "selection"), "selection.tie_break"),
        level=_require_str(_required(raw, "level", "selection"), "selection.level"),
        aggregation=_require_str(_required(raw, "aggregation", "selection"), "selection.aggregation"),
        warmup_from=_require_str(_required(raw, "warmup_from", "selection"), "selection.warmup_from"),
    )


def _parse_evaluation(raw: dict[str, Any]) -> EvaluationConfig:
    allowed = {"primary_metric", "secondary_metric", "aggregation"}
    _reject_unknown_keys(raw, allowed, "evaluation")
    return EvaluationConfig(
        primary_metric=_require_str(_required(raw, "primary_metric", "evaluation"), "evaluation.primary_metric"),
        secondary_metric=_require_str(
            _required(raw, "secondary_metric", "evaluation"), "evaluation.secondary_metric"
        ),
        aggregation=_require_str(_required(raw, "aggregation", "evaluation"), "evaluation.aggregation"),
    )


def _parse_bootstrap(raw: dict[str, Any]) -> BootstrapConfig:
    allowed = {"method", "block_months", "block_sensitivity", "replicates", "confidence", "seed"}
    _reject_unknown_keys(raw, allowed, "bootstrap")
    block_months = _require_positive_int(
        _required(raw, "block_months", "bootstrap"), "bootstrap.block_months"
    )
    seed = _require_int(_required(raw, "seed", "bootstrap"), "bootstrap.seed")
    return BootstrapConfig(
        method=_require_str(_required(raw, "method", "bootstrap"), "bootstrap.method"),
        block_months=block_months,
        block_sensitivity=_require_sensitivity(
            _required(raw, "block_sensitivity", "bootstrap"), "bootstrap.block_sensitivity", block_months
        ),
        replicates=_require_positive_int(_required(raw, "replicates", "bootstrap"), "bootstrap.replicates"),
        confidence=_require_float(_required(raw, "confidence", "bootstrap"), "bootstrap.confidence"),
        seed=seed,
    )


def _require_float(value: Any, path: str) -> float:
    _require_type(value, float, path)
    if not 0.0 < value < 1.0:
        raise ConfigError(f"Параметр '{path}' должен лежать в интервале (0, 1), получено {value}")
    return value


def _required(section: dict[str, Any], key: str, section_name: str) -> Any:
    _require_key(section, key, section_name)
    return section[key]


# --- сборка и проверка согласованности ----------------------------------------


def _parse_study(data: dict[str, Any]) -> StudyConfig:
    allowed_sections = {
        "protocol",
        "period",
        "universe",
        "features",
        "month_end",
        "model",
        "selection",
        "evaluation",
        "bootstrap",
    }
    unknown_sections = set(data) - allowed_sections
    if unknown_sections:
        raise ConfigError(f"Неизвестные секции конфигурации: {', '.join(sorted(unknown_sections))}")

    for section in allowed_sections:
        if section not in data:
            raise ConfigError(f"Отсутствует обязательная секция '{section}'")

    config = StudyConfig(
        protocol=_parse_protocol(data["protocol"]),
        period=_parse_period(data["period"]),
        universe=_parse_universe(data["universe"]),
        features=_parse_features(data["features"]),
        month_end=_parse_month_end(data["month_end"]),
        model=_parse_model(data["model"]),
        selection=_parse_selection(data["selection"]),
        evaluation=_parse_evaluation(data["evaluation"]),
        bootstrap=_parse_bootstrap(data["bootstrap"]),
    )
    _check_consistency(config)
    return config


def _check_consistency(config: StudyConfig) -> None:
    period = config.period
    bootstrap = config.bootstrap
    model = config.model
    selection = config.selection

    # Дата начала загрузки предшествует началу наблюдения не менее чем на 6 мес.
    download_lag_months = _months_between(period.download_start, period.observation_start)
    if download_lag_months < 6:
        raise ConfigError(
            "Параметры 'period.observation_start' и 'period.download_start' несогласованы: "
            f"загрузка должна начинаться не менее чем за 6 месяцев до наблюдения "
            f"(реальный лаг {download_lag_months} мес., download_start={period.download_start}, "
            f"observation_start={period.observation_start})"
        )

    # Конец периода разработки предшествует началу периода оценки.
    if period.development_target_end >= period.evaluation_target_start:
        raise ConfigError(
            "Параметры 'period.development_target_end' и 'period.evaluation_target_start' "
            "несогласованы: конец разработки должен предшествовать началу оценки "
            f"({period.development_target_end} >= {period.evaluation_target_start})"
        )

    # Длина проверочного периода при отборе не превышает длину периода оценки.
    evaluation_months = _months_between(period.evaluation_target_start, period.observation_end)
    if selection.selection_window_months > evaluation_months:
        raise ConfigError(
            "Параметры 'selection.selection_window_months' и период оценки несогласованы: "
            f"окно отбора ({selection.selection_window_months} мес.) больше периода оценки "
            f"({evaluation_months} мес.)"
        )

    # Зерно генератора положительное.
    if bootstrap.seed <= 0:
        raise ConfigError(
            f"Параметр 'bootstrap.seed' должен быть положительным целым, получено {bootstrap.seed}"
        )

    # Окно обучения наполняется данными (Б.5): число завершённых полных пар
    # за весь период наблюдения не меньше длины окна обучения.
    # Первая полная пара требует признаков месяца s0 и доходности месяца s0+1;
    # последняя — признаков месяца, предшествующего observation_end. Число таких
    # пар равно числу месяцев от s0 до observation_end.
    s0 = config.first_observation_month
    available_pairs = _months_between(s0, period.observation_end)
    if available_pairs < model.training_window_months:
        raise ConfigError(
            "Параметры 'model.training_window_months' и период наблюдения несогласованы: "
            f"за период наблюдения доступно {available_pairs} завершённых полных пар, "
            f"а требуется {model.training_window_months}"
        )


def _months_between(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month)


def load_config(path: str | Path) -> StudyConfig:
    """Загрузить и провалидировать конфигурацию из TOML-файла."""
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Файл конфигурации не найден: {config_path.resolve()}")

    if config_path.suffix.lower() != ".toml":
        raise ConfigError(f"Ожидался файл конфигурации с расширением .toml: {config_path}")

    try:
        with config_path.open("rb") as file:
            data = tomllib.load(file)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Некорректный TOML в {config_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError("Корень конфигурации должен быть TOML-таблицей (объектом)")

    return _parse_study(data)