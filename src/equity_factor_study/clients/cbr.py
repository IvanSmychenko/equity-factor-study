"""Клиент Банка России: дневной ряд ключевой ставки.

Метод ``KeyRateXML`` веб-сервиса DailyInfo (SOAP 1.1) возвращает дневные
значения ключевой ставки в процентах годовых. Ответ разбирается безопасным
парсером (``defusedxml``) без исполнения кода; SOAP Fault преобразуется в
отдельное исключение ``CBRFaultError``.

Структура ответа установлена по фактическому ответу источника и зафиксирована
в документации: ряд представлен элементами ``KeyRate``/``KR``, поле даты — ``DT``
(dateTime с временной зоной), значение ставки — ``Rate`` (строка, проценты
годовых).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
from defusedxml import ElementTree as SafeET

from equity_factor_study.http_client import HttpClient
from equity_factor_study.storage import save_response

# Адрес и пространство имён веб-сервиса DailyInfo.
CBR_ENDPOINT = "https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx"
CBR_NAMESPACE = "http://web.cbr.ru/"
CBR_ACTION = CBR_NAMESPACE + "KeyRateXML"

# Теги ответа, установленные по фактическому ответу источника.
_RESPONSE_TAG = "KeyRateXMLResponse"
_RESULT_TAG = "KeyRateXMLResult"
_KEYRATE_TAG = "KeyRate"
_ROW_TAG = "KR"
_DATE_TAG = "DT"
_RATE_TAG = "Rate"

# Единица измерения значений ряда.
RATE_UNIT = "percent_per_annum"


class CBRFaultError(Exception):
    """SOAP Fault от сервера Банка России."""


class CBRResponseError(Exception):
    """Ответ не соответствует ожидаемой структуре KeyRateXML."""


@dataclass
class KeyRateDownload:
    table: pd.DataFrame
    raw_path: Path
    unit: str

    @property
    def rows(self) -> int:
        return len(self.table)


def _local_name(tag: str) -> str:
    """Вернуть локальное имя XML-тега без учёта пространства имён."""
    return tag.rsplit("}", 1)[-1]


def _soap_envelope(from_date: date, to_date: date) -> bytes:
    """Собрать SOAP 1.1-конверт для запроса KeyRateXML."""
    envelope = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<soap:Envelope '
        'xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xmlns:xsd="http://www.w3.org/2001/XMLSchema">'
        "<soap:Body>"
        f'<KeyRateXML xmlns="{CBR_NAMESPACE}">'
        f"<fromDate>{from_date.isoformat()}</fromDate>"
        f"<ToDate>{to_date.isoformat()}</ToDate>"
        "</KeyRateXML>"
        "</soap:Body>"
        "</soap:Envelope>"
    )
    return envelope.encode("utf-8")


def _soap_headers() -> dict[str, str]:
    """Заголовки HTTP-запроса, требуемые протоколом SOAP 1.1."""
    return {
        "Content-Type": "text/xml; charset=utf-8",
        "SOAPAction": CBR_ACTION,
    }


def _raise_on_fault(root) -> None:
    """Обнаружить SOAP Fault и поднять ``CBRFaultError`` с текстом ответа."""
    for element in root.iter():
        if _local_name(element.tag) == "Fault":
            message_parts: list[str] = []
            for child in element.iter():
                name = _local_name(child.tag)
                if name in {"faultstring", "faultcode", "reason"}:
                    fragment = (child.text or "").strip()
                    if fragment:
                        message_parts.append(f"{name}: {fragment}")
            raise CBRFaultError(" | ".join(message_parts) or "неизвестная ошибка SOAP Fault")


def _validate_structure(root) -> None:
    """Проверить, что ответ имеет ожидаемую структуру KeyRateXML."""
    expected = (_RESPONSE_TAG, _RESULT_TAG, _KEYRATE_TAG)
    stack: list[str] = []
    for element in root.iter():
        name = _local_name(element.tag)
        if name in expected or name == _ROW_TAG:
            stack.append(name)
    # Для пустого периода нет элементов KR — это допустимо, проверяем
    # обязательную обёртку KeyRateXMLResponse.
    if _RESPONSE_TAG not in stack:
        raise CBRResponseError(
            f"Ответ не содержит {_RESPONSE_TAG!r}:"
            f" найденные элементы — {[n for n in stack[:10]] or 'нет'}"
        )


def parse_key_rate(xml: str | bytes) -> pd.DataFrame:
    """Разобрать ответ KeyRateXML в таблицу с колонками ``date`` и ``rate``.

    Возвращает DataFrame с колонками:
    - ``date`` — дата (datetime.date) последнего торгового дня;
    - ``rate`` — значение ключевой ставки (float, процентов годовых).
    Строки упорядочиваются по дате по возрастанию.
    """
    root = SafeET.fromstring(xml)
    _raise_on_fault(root)
    _validate_structure(root)

    rows: list[tuple[date, float]] = []
    for element in root.iter():
        if _local_name(element.tag) != _ROW_TAG:
            continue
        dt_raw: str | None = None
        rate_raw: str | None = None
        for child in element.iter():
            child_name = _local_name(child.tag)
            if child_name == _DATE_TAG:
                dt_raw = (child.text or "").strip()
            elif child_name == _RATE_TAG:
                rate_raw = (child.text or "").strip()

        if dt_raw is None or rate_raw is None:
            raise CBRResponseError("Строка KR не содержит полей DT и/или Rate")

        # DT имеет вид 2014-01-31T00:00:00+04:00 — берём дату без времени.
        parsed_dt = datetime.fromisoformat(dt_raw)
        # ValueError при некорректном значении является достаточным сигналом.
        value = float(rate_raw)
        rows.append((parsed_dt.date(), value))

    # Дубликаты дат недопустимы в дневном ряду.
    if len(rows) != len({d for d, _ in rows}):
        raise CBRResponseError("Дневной ряд содержит повторяющиеся даты")

    table = pd.DataFrame(rows, columns=["date", "rate"]).sort_values("date")
    table = table.reset_index(drop=True)
    table["date"] = pd.to_datetime(table["date"]).dt.date
    return table


class CBRClient:
    """Клиент веб-сервиса DailyInfo Банка России."""

    def __init__(self, http: HttpClient, raw_dir: Path) -> None:
        self.http = http
        self.raw_dir = raw_dir

    def get_key_rate(self, *, from_date: date, to_date: date) -> KeyRateDownload:
        """Запросить ряд ключевой ставки за период и сохранить исходный ответ."""
        if from_date > to_date:
            raise ValueError("from_date не может быть позже to_date")

        response = self.http.post(
            CBR_ENDPOINT,
            data=_soap_envelope(from_date, to_date),
            headers=_soap_headers(),
        )
        retrieved_at = datetime.now(UTC).isoformat()

        file_name = f"keyrate_{from_date.isoformat()}_{to_date.isoformat()}.xml"
        raw_path = save_response(
            output_dir=self.raw_dir,
            name=file_name,
            content=response.content,
            metadata={
                "source": "CBR DailyInfo",
                "request_method": response.request.method,
                "url": response.url,
                "soap_action": CBR_ACTION,
                "request_parameters": {
                    "fromDate": from_date.isoformat(),
                    "ToDate": to_date.isoformat(),
                },
                "retrieved_at_utc": retrieved_at,
                "status_code": response.status_code,
                "content_type": response.headers.get("Content-Type"),
            },
        )

        table = parse_key_rate(response.content)
        return KeyRateDownload(table=table, raw_path=raw_path, unit=RATE_UNIT)