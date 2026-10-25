from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from equity_factor_study.http import HttpClient
from equity_factor_study.storage import save_response


@dataclass
class HistoryPage:
    table: pd.DataFrame
    cursor: pd.DataFrame
    raw_path: Path


@dataclass
class SecurityCatalog:
    table: pd.DataFrame
    raw_path: Path

    def __bool__(self) -> bool:
        return not self.table.empty

    @property
    def rows(self) -> int:
        return len(self.table)


@dataclass
class HistoryDownload:
    table: pd.DataFrame
    raw_paths: list[Path]
    page_sizes: list[int]
    cursor_total: int | None

    def __bool__(self) -> bool:
        return not self.table.empty

    @property
    def pages(self) -> int:
        return len(self.raw_paths)

    @property
    def rows(self) -> int:
        return len(self.table)

    @property
    def date_interval(self) -> tuple[date | None, date | None]:
        """Реальный интервал торговых дат, наблюдаемый в данных."""
        if self.table.empty or "TRADEDATE" not in self.table.columns:
            return None, None
        trade_dates = pd.to_datetime(self.table["TRADEDATE"], errors="coerce").dropna()
        if trade_dates.empty:
            return None, None
        return trade_dates.min().date(), trade_dates.max().date()


class MOEXHistoryError(Exception):
    """Ошибка сбора истории MOEX (непродвижение курсора, дубли, неполнота)."""


def _cursor_values(cursor: pd.DataFrame) -> tuple[int | None, int | None, int | None]:
    """Извлечь (index, total, page_size) из строки курсора."""
    if cursor.empty:
        return None, None, None
    row = cursor.iloc[0]
    index = int(row["INDEX"]) if "INDEX" in cursor.columns else None
    total = int(row["TOTAL"]) if "TOTAL" in cursor.columns else None
    page_size = int(row["PAGESIZE"]) if "PAGESIZE" in cursor.columns else None
    return index, total, page_size


def parse_block(payload: dict, name: str) -> pd.DataFrame:
    """Преобразовать стандартный табличный блок ISS."""
    if name not in payload:
        raise ValueError(f"В ответе ISS отсутствует блок {name!r}")

    block = payload[name]

    if not isinstance(block, dict):
        raise TypeError(f"Блок {name!r} должен быть JSON-объектом")

    columns = block.get("columns")
    rows = block.get("data")

    if not isinstance(columns, list) or not isinstance(rows, list):
        raise TypeError(f"Поля columns и data блока {name!r} должны быть списками")

    if not all(isinstance(column, str) for column in columns):
        raise TypeError(f"Названия столбцов в {name!r} должны быть строками")

    if len(columns) != len(set(columns)):
        raise ValueError(f"Повторяющиеся столбцы в {name!r}")

    if not all(isinstance(row, list) for row in rows):
        raise TypeError(f"Строки блока {name!r} должны быть списками")

    if not all(len(row) == len(columns) for row in rows):
        raise ValueError(f"Число значений не соответствует столбцам в {name!r}")

    return pd.DataFrame(rows, columns=columns)


class MOEXClient:
    BASE_URL = "https://iss.moex.com"

    def __init__(self, http: HttpClient, raw_dir: Path) -> None:
        self.http = http
        self.raw_dir = raw_dir

    def get_history_page(
        self,
        *,
        security: str,
        from_date: date,
        till_date: date,
        board: str = "TQBR",
        start: int = 0,
    ) -> HistoryPage:
        if not security.isascii() or not security.isalnum():
            raise ValueError("security должен содержать латинские буквы и цифры")

        if not board.isascii() or not board.isalnum():
            raise ValueError("board должен содержать латинские буквы и цифры")

        if from_date > till_date:
            raise ValueError("from_date не может быть позже till_date")

        if start < 0:
            raise ValueError("start не может быть отрицательным")

        url = (
            f"{self.BASE_URL}/iss/history/engines/stock/markets/shares/"
            f"boards/{board}/securities/{security}.json"
        )

        params = {
            "from": from_date.isoformat(),
            "till": till_date.isoformat(),
            "start": start,
            "iss.only": "history,history.cursor",
        }

        response = self.http.get(url, params=params)
        retrieved_at = datetime.now(UTC).isoformat()

        file_name = (
            f"{security}_{board}_{from_date.isoformat()}_{till_date.isoformat()}_{start:06d}.json"
        )

        raw_path = save_response(
            output_dir=self.raw_dir,
            name=file_name,
            content=response.content,
            metadata={
                "source": "MOEX ISS",
                "request_method": response.request.method,
                "url": response.url,
                "request_parameters": params,
                "retrieved_at_utc": retrieved_at,
                "status_code": response.status_code,
                "content_type": response.headers.get("Content-Type"),
            },
        )

        # Исходный ответ уже сохранён, даже если разбор завершится ошибкой.
        payload = response.json()

        if not isinstance(payload, dict):
            raise TypeError("Ожидался JSON-объект ISS")

        table = parse_block(payload, "history")

        if "history.cursor" in payload:
            cursor = parse_block(payload, "history.cursor")
        else:
            cursor = pd.DataFrame()

        return HistoryPage(
            table=table,
            cursor=cursor,
            raw_path=raw_path,
        )

    def get_securities(
        self,
        *,
        board: str = "TQBR",
        start: int = 0,
    ) -> SecurityCatalog:
        """Получить перечень инструментов, торгуемых на режиме торгов.

        Возвращает справочную таблицу (SECID, SECTYPE, SECNAME, LISTLEVEL и др.)
        по текущему состоянию режима. Отдельное сохранение ответа выполняется
        как для истории: исходный JSON и метаданные.
        """
        url = (
            f"{self.BASE_URL}/iss/engines/stock/markets/shares/"
            f"boards/{board}/securities.json"
        )
        params = {
            "start": start,
            "iss.only": "securities",
        }
        response = self.http.get(url, params=params)
        retrieved_at = datetime.now(UTC).isoformat()

        file_name = f"{board}_securities_{start:06d}.json"
        raw_path = save_response(
            output_dir=self.raw_dir,
            name=file_name,
            content=response.content,
            metadata={
                "source": "MOEX ISS",
                "request_method": response.request.method,
                "url": response.url,
                "request_parameters": params,
                "retrieved_at_utc": retrieved_at,
                "status_code": response.status_code,
                "content_type": response.headers.get("Content-Type"),
            },
        )

        payload = response.json()
        if not isinstance(payload, dict) or "securities" not in payload:
            raise ValueError("В ответе ISS отсутствует блок 'securities'")
        table = parse_block(payload, "securities")
        return SecurityCatalog(table=table, raw_path=raw_path)

    def get_full_history(
        self,
        *,
        security: str,
        from_date: date,
        till_date: date,
        board: str = "TQBR",
        start: int = 0,
    ) -> HistoryDownload:
        """Собрать полную историю по инструменту за интервал.

        Итерация продолжается до пустой страницы либо достижения общего числа
        записей из курсора; размер страницы берётся из курсора. По завершении
        проверяются отсутствие дублей по паре «инструмент — дата» и полнота
        относительно курсора. При непродвижении позиции сбор прерывается.
        """
        pages: list[pd.DataFrame] = []
        raw_paths: list[Path] = []
        page_sizes: list[int] = []
        cursor_total: int | None = None

        current = start
        last_index: int | None = None
        while True:
            page = self.get_history_page(
                security=security,
                from_date=from_date,
                till_date=till_date,
                board=board,
                start=current,
            )

            if page.table.empty:
                break

            index, total, page_size = _cursor_values(page.cursor)
            if total is not None:
                cursor_total = total

            # Размер страницы берётся из курсора; если курсора нет — по факту.
            effective_page_size = page_size if page_size else len(page.table)
            if effective_page_size <= 0:
                raise MOEXHistoryError(
                    f"Курсор {security}@{board} вернул неположительный размер страницы: "
                    f"{effective_page_size}"
                )

            page_sizes.append(len(page.table))
            pages.append(page.table)
            raw_paths.append(page.raw_path)

            # Позиция следующей страницы определяется курсором (INDEX), а если
            # курсора нет — по размеру текущей страницы.
            next_position = (index if index is not None else current) + effective_page_size

            # Защита от зацикливания: позиция должна продвинуться.
            if next_position <= current:
                raise MOEXHistoryError(
                    f"Курсор {security}@{board} не продвигает позицию (start={current}, "
                    f"index={index}, page_size={effective_page_size})"
                )
            if index is not None and last_index is not None and index <= last_index:
                raise MOEXHistoryError(
                    f"Курсор {security}@{board} не продвигает позицию (index вернулся "
                    f"к {index} с {last_index})"
                )

            if total is not None and next_position >= total:
                break

            last_index = index if index is not None else current
            current = next_position

        if not pages:
            return HistoryDownload(
                table=pd.DataFrame(),
                raw_paths=raw_paths,
                page_sizes=page_sizes,
                cursor_total=cursor_total,
            )

        table = pd.concat(pages, ignore_index=True)
        self._validate_full_history(table, cursor_total, security=security, board=board)

        return HistoryDownload(
            table=table,
            raw_paths=raw_paths,
            page_sizes=page_sizes,
            cursor_total=cursor_total,
        )

    @staticmethod
    def _validate_full_history(
        table: pd.DataFrame,
        cursor_total: int | None,
        *,
        security: str,
        board: str,
    ) -> None:
        """Проверить собранную историю на дубли и полноту относительно курсора."""
        if "SECID" in table.columns and "TRADEDATE" in table.columns:
            duplicates = table.duplicated(subset=["SECID", "TRADEDATE"])
            if duplicates.any():
                duplicated = table.loc[duplicates, ["SECID", "TRADEDATE"]].drop_duplicates()
                examples = ", ".join(
                    f"{row.SECID}@{row.TRADEDATE}" for row in duplicated.head(5).itertuples()
                )
                raise MOEXHistoryError(
                    f"Найдены дубли по паре «инструмент — дата» для {security}@{board}: {examples}"
                )

        if cursor_total is not None and len(table) != cursor_total:
            raise MOEXHistoryError(
                f"Неполная история {security}@{board}: собрано {len(table)} строк, "
                f"курсор сообщает {cursor_total}"
            )
