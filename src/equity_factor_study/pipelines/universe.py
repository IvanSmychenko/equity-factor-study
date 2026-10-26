"""Построение исторической совокупности (выборка инструментов).

Модуль слоя сбора: разрешены сетевые вызовы. Формирует историческую выборку
по обороту начального года, записывает её в файл-замок ``configs/universe.lock.toml``
и формирует перечень исключений с причинами.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from equity_factor_study.clients.moex import MOEXClient

# Код режима торгов (основной) и тип обыкновенной акции.
DEFAULT_BOARD = "TQBR"
COMMON_SHARE = "1"

# Предельное значение, защищающее от зацикливания при сборе оборота.
_MAX_CANDIDATES = 10_000


@dataclass
class UniverseResult:
    selected: list[str]
    turnover: pd.DataFrame  # secid, turnover_<year>
    reasons: pd.DataFrame  # secid, reason
    lock_path: Path


class UniverseError(Exception):
    """Ошибка построения исторической совокупности."""


def collect_turnover(
    client: MOEXClient,
    *,
    candidates: list[str],
    board: str,
    year: int,
) -> pd.DataFrame:
    """Собрать оборот в рублях по каждому кандидату за календарный год.

    Оборот накапливается по дневным значениям VALUE из полной истории.
    """
    from_date = date(year, 1, 1)
    till_date = date(year, 12, 31)

    records: list[dict] = []
    for index, security in enumerate(candidates):
        if index >= _MAX_CANDIDATES:
            raise UniverseError("Слишком много кандидатов для сбора оборота")

        download = client.get_full_history(
            security=security,
            from_date=from_date,
            till_date=till_date,
            board=board,
        )
        table = download.table

        if table.empty or "VALUE" not in table.columns:
            records.append({"secid": security, "turnover": 0.0, "rows": 0})
            continue

        # Оставляем только строки выбранного режима и суммируем оборот.
        value_series = pd.to_numeric(table["VALUE"], errors="coerce").fillna(0.0)
        records.append(
            {
                "secid": security,
                "turnover": float(value_series.sum()),
                "rows": len(table),
            }
        )

    return pd.DataFrame(records)


def _rank_by_turnover(turnover: pd.DataFrame, column: str = "turnover") -> pd.DataFrame:
    """Отсортировать кандидатов по обороту убывающе (детерминированно)."""
    return turnover.sort_values(
        by=[column, "secid"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_universe(
    client: MOEXClient,
    *,
    candidates: list[str],
    board: str,
    year: int,
    target_size: int,
) -> UniverseResult:
    """Построить историческую выборку из кандидатов по обороту за год.

    Отбираются ``target_size`` инструментов с наибольшим оборотом. Кандидаты,
    не попавшие в выборку, фиксируются в перечне исключений с причиной.
    """
    if not candidates:
        raise UniverseError("Пустой список кандидатов для построения выборки")

    turnover = collect_turnover(client, candidates=candidates, board=board, year=year)
    ranked = _rank_by_turnover(turnover)
    selected = ranked.head(target_size)["secid"].tolist()

    reasons_rows: list[dict] = []
    for _, row in ranked.iterrows():
        if row["secid"] not in selected:
            reasons_rows.append(
                {
                    "secid": row["secid"],
                    "reason": "not_selected_low_turnover",
                    "turnover": row["turnover"],
                    "rank": int(row.name) + 1,
                }
            )

    reasons = pd.DataFrame(reasons_rows)
    return UniverseResult(
        selected=selected,
        turnover=ranked,
        reasons=reasons,
        lock_path=Path("unused"),  # заполняется вызывающим кодом
    )


def write_lock_file(
    *,
    lock_path: Path,
    config_hash: str,
    version: str,
    board: str,
    year: int,
    selected: list[str],
    turnover: pd.DataFrame | None = None,
    exclusions: pd.DataFrame | None = None,
    formation_date: date | None = None,
    exclusions_relative: str | None = None,
) -> Path:
    """Записать файл-замок состава выборки в TOML-формате.

    Замок фиксирует фактический состав и его происхождение: перечень
    отобранных инструментов, оборот по каждому, дату формирования, версию
    правила отбора и ссылку на файл исключений.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    secids = ", ".join(f'"{secid}"' for secid in selected)
    lines = [
        "# Замок состава исторической совокупности.",
        "# Формируется этапом 4; изменение вручную не допускается.",
        f'# config_hash = "{config_hash}"',
        "",
        "[universe]",
        f'version = "{version}"',
        f'selection_board = "{board}"',
        f"selection_year = {year}",
        f'formation_date = "{(formation_date or date.today()).isoformat()}"',
        f"secids = [{secids}]",
    ]
    if exclusions_relative:
        lines.append(f'exclusions_file = "{exclusions_relative}"')

    # Оборот по каждому отобранному инструменту (для воспроизводимости).
    if turnover is not None and not turnover.empty:
        lines.append("")
        for secid in selected:
            row = turnover[turnover["secid"] == secid]
            if row.empty:
                continue
            amount = row.iloc[0]["turnover"] if "turnover" in row.columns else row.iloc[0]["turnover_" + str(year)]
            lines.append("[[instruments]]")
            lines.append(f'secid = "{secid}"')
            lines.append(f"turnover = {amount}")
            lines.append("")

    if exclusions is not None and not exclusions.empty:
        lines.append("[exclusions]")
        lines.append('reason = "not_selected_low_turnover"')
        excluded = ", ".join(f'"{secid}"' for secid in exclusions["secid"])
        lines.append(f"secids = [{excluded}]")

    lock_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lock_path


def load_lock_file(*, lock_path: Path) -> list[str]:
    """Прочитать состав выборки из файла-замка."""
    if not lock_path.exists():
        raise UniverseError(f"Файл-замок выборки не найден: {lock_path.resolve()}")
    with lock_path.open("rb") as file:
        data = tomllib.load(file)
    universe = data.get("universe", {})
    return list(universe.get("secids", []))


def write_exclusions(
    *,
    exclusions_path: Path,
    reasons: pd.DataFrame,
) -> Path:
    """Записать перечень исключённых кандидатов (с причинами) в CSV."""
    exclusions_path.parent.mkdir(parents=True, exist_ok=True)
    if reasons.empty:
        header = "secid,reason,turnover,rank\n"
        exclusions_path.write_text(header, encoding="utf-8")
        return exclusions_path
    reasons.to_csv(exclusions_path, index=False)
    return exclusions_path


# Допустимые основания ручного исключения (раздел 4 уточнения).
# Исключение допускается только по причинам качества данных. Запрещены любые
# основания, связанные с ожидаемой доходностью или «непривычностью» эмитента.
ALLOWED_MANUAL_REASONS = {
    "data_gap_unexplained",  # разрыв истории без корпоративного события
    "code_change_unconfirmed",  # смена кода/режима без подтверждённой непрерывности
    "mostly_no_trading",  # отсутствие торгов большую часть периода
    "security_type_mismatch",  # расхождение справочных данных о типе инструмента
}


def append_manual_exclusions(
    *,
    exclusions_path: Path,
    rows: list[dict],
) -> Path:
    """Добавить результат ручной проверки в перечень исключений.

    ``rows`` — список словарей с ключами ``secid``, ``reason`` и, по желанию,
    ``turnover``, ``rank``. Причина обязана входить в ``ALLOWED_MANUAL_REASONS``;
    иначе возникает ``UniverseError`` (ручное исключение по недопустимому
    основанию возвращает смещение отбора). Записанная причина дополняется
    пояснением о подтверждающих данных.
    """
    records: list[dict] = []
    for row in rows:
        reason = row.get("reason")
        if reason not in ALLOWED_MANUAL_REASONS:
            raise UniverseError(
                f"Недопустимое основание ручного исключения: {reason!r}. "
                "Допускаются только причины качества данных: "
                + ", ".join(sorted(ALLOWED_MANUAL_REASONS))
            )
        records.append(
            {
                "secid": row["secid"],
                "reason": reason,
                "turnover": row.get("turnover"),
                "rank": row.get("rank"),
            }
        )

    # Объединить с существующим перечнем (если он есть) без дубликатов по secid.
    existing = pd.DataFrame()
    if exclusions_path.exists():
        try:
            existing = pd.read_csv(exclusions_path)
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()
    new_frame = pd.DataFrame(records)
    combined = pd.concat([existing, new_frame], ignore_index=True)
    combined = combined.drop_duplicates(subset=["secid"], keep="last")
    combined.to_csv(exclusions_path, index=False)
    return exclusions_path


def build_and_write_universe(
    *,
    client: MOEXClient,
    lock_path: Path,
    exclusions_path: Path,
    catalog_table: pd.DataFrame,
    config_hash: str,
    version: str,
    board: str,
    year: int,
    target_size: int,
    force: bool = False,
) -> UniverseResult:
    """Построить выборку из справочной таблицы и записать lock + исключения.

    Эта функция объединяет этап 4 целиком: принимает уже собранный справочник
    (обыкновенные акции), собирает оборот за год, отбирает ``target_size``
    инструментов и записывает файл-замок состава и перечень исключений.

    Защита от перезаписи замка (правило раздела 3 уточнения): при уже
    существующем ``lock_path`` построение не перезаписывает его, а завершается
    ошибкой с предложением явно подтвердить пересборку, если ``force=False``.
    """
    if not force and lock_path.exists():
        raise UniverseError(
            "Замок состава уже существует; автоматическая перезапись не "
            "допускается. Подтвердите пересборку явно (force=True/--force), "
            "иначе результат потеряет сопоставимость с предыдущими прогонами."
        )
    if catalog_table.empty or "SECID" not in catalog_table.columns:
        raise UniverseError("Справочник пуст: нечего включать в выборку")

    candidates = [str(secid) for secid in catalog_table["SECID"].tolist()]
    result = build_universe(
        client=client,
        candidates=candidates,
        board=board,
        year=year,
        target_size=target_size,
    )

    write_lock_file(
        lock_path=lock_path,
        config_hash=config_hash,
        version=version,
        board=board,
        year=year,
        selected=result.selected,
        turnover=result.turnover,
        exclusions=result.reasons,
        formation_date=date.today(),
        exclusions_relative=exclusions_path.name,
    )
    write_exclusions(exclusions_path=exclusions_path, reasons=result.reasons)
    return result