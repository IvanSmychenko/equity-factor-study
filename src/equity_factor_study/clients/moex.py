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
