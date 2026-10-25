"""Справочники инструментов и режимов торгов.

Модуль слоя сбора: разрешены сетевые вызовы. Формирует перечень обыкновенных
акций, обращающихся на основном режиме торгов, и сохраняет исходные ответы в
каталог выгрузки.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from equity_factor_study.clients.moex import MOEXClient, SecurityCatalog

# Код типа бумаги обыкновенной акции в справочнике ISS.
SECURITY_TYPE_COMMON_SHARE = "1"

# Словарь семантических меток (из configs/study.toml) в коды ISS.
SECURITY_TYPE_LABELS = {
    "common_share": SECURITY_TYPE_COMMON_SHARE,
}


def resolve_security_type(security_type: str) -> str:
    """Нормализовать тип бумаги в код ISS.

    Конфигурация описывает тип бумаги семантической меткой (например,
    ``common_share``), тогда как MOEX использует числовые коды (``1``).
    Функция приводит метку к коду, а числовой код оставляет без изменений;
    неизвестная метка не принимается, чтобы ошибка конфигурации была явной.
    """
    resolved = SECURITY_TYPE_LABELS.get(security_type)
    if resolved is not None:
        return resolved
    if security_type in SECURITY_TYPE_LABELS.values():
        return security_type
    raise ValueError(f"Неизвестный тип бумаги: {security_type!r}")


@dataclass
class CatalogResult:
    common_shares: pd.DataFrame
    table: pd.DataFrame
    raw_paths: list[Path]


def load_catalog(
    client: MOEXClient,
    *,
    board: str,
    security_type: str = SECURITY_TYPE_COMMON_SHARE,
) -> CatalogResult:
    """Загрузить справочник режима и отфильтровать обыкновенные акции.

    Справочник может занимать несколько страниц (размер берётся из курсора
    при его наличии). Возвращаются и полный справочник, и отфильтрованная
    таблица обыкновенных акций.
    """
    tables: list[pd.DataFrame] = []
    raw_paths: list[Path] = []
    start = 0
    while True:
        page: SecurityCatalog = client.get_securities(board=board, start=start)
        table = page.table
        raw_paths.append(page.raw_path)

        if table.empty:
            break

        tables.append(table)

        # Позиция следующей страницы — по фактическому числу строк на странице.
        next_start = start + len(table)
        if next_start <= start:
            break
        start = next_start

    if not tables:
        empty = pd.DataFrame()
        return CatalogResult(common_shares=empty, table=empty, raw_paths=raw_paths)

    full = pd.concat(tables, ignore_index=True)
    common = _filter_common_shares(full, resolve_security_type(security_type))
    return CatalogResult(common_shares=common, table=full, raw_paths=raw_paths)


def _filter_common_shares(table: pd.DataFrame, security_type: str) -> pd.DataFrame:
    """Оставить только обыкновенные акции (по типу и режиму торгов)."""
    if table.empty:
        return table

    filtered = table
    if "SECTYPE" in table.columns:
        filtered = filtered[filtered["SECTYPE"] == security_type]
    if "SECID" in table.columns:
        filtered = filtered.dropna(subset=["SECID"])
        filtered = filtered[filtered["SECID"] != ""]

    return filtered.reset_index(drop=True)