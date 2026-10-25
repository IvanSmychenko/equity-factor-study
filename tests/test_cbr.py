"""Тесты клиента Банка России: разбор KeyRateXML без обращения к сети."""

from datetime import date
from pathlib import Path
from xml.etree.ElementTree import ParseError

import pandas as pd
import pytest
import responses

from equity_factor_study.clients.cbr import (
    CBR_ENDPOINT,
    CBRClient,
    CBRFaultError,
    CBRResponseError,
    KeyRateDownload,
    parse_key_rate,
)
from equity_factor_study.http import HttpClient

FIXTURES = Path(__file__).resolve().parent / "fixtures"

FIXTURE_OK = FIXTURES / "cbr_keyrate_2014-01.xml"
FIXTURE_EMPTY = FIXTURES / "cbr_keyrate_empty.xml"
FIXTURE_FAULT = FIXTURES / "cbr_keyrate_fault.xml"


def _read(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def test_parse_key_rate_ok():
    """Успешный разбор сохранённого образца даёт упорядоченный ряд."""
    table = parse_key_rate(_read("cbr_keyrate_2014-01.xml"))
    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == ["date", "rate"]
    assert table["date"].iloc[0] == date(2014, 1, 9)
    assert table["date"].iloc[-1] == date(2014, 1, 31)
    assert table["rate"].iloc[0] == pytest.approx(5.50)
    # Ряд упорядочен по возрастанию дат.
    assert table["date"].is_monotonic_increasing
    assert len(table) == table["date"].nunique()


def test_parse_key_rate_empty_period():
    """Пустой период даёт пустую таблицу без ошибок."""
    table = parse_key_rate(_read("cbr_keyrate_empty.xml"))
    assert table.empty
    assert list(table.columns) == ["date", "rate"]


def test_parse_key_rate_soap_fault_raises():
    """SOAP Fault преобразуется в отдельное исключение с текстом из ответа."""
    with pytest.raises(CBRFaultError) as exc_info:
        parse_key_rate(_read("cbr_keyrate_fault.xml"))
    assert "Неверный формат даты" in str(exc_info.value)


def test_parse_key_rate_invalid_xml_raises():
    """Некорректный XML завершается ошибкой разбора без исполнения кода."""
    with pytest.raises(ParseError):
        parse_key_rate(b"<not-closed")


def test_parse_key_rate_wrong_structure_raises():
    """Ответ без структуры KeyRateXML отвергается."""
    xml = (
        '<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">'
        "<soap:Body><Other/></soap:Body></soap:Envelope>"
    )
    with pytest.raises(CBRResponseError):
        parse_key_rate(xml)


def test_parse_key_rate_duplicate_dates_raises():
    """Повторяющиеся даты в дневном ряду отвергаются."""
    xml = (
        "<KeyRateXMLResponse><KeyRateXMLResult><KeyRate>"
        "<KR><DT>2014-01-31T00:00:00+04:00</DT><Rate>5.50</Rate></KR>"
        "<KR><DT>2014-01-31T00:00:00+04:00</DT><Rate>5.55</Rate></KR>"
        "</KeyRate></KeyRateXMLResult></KeyRateXMLResponse>"
    )
    with pytest.raises(CBRResponseError, match="повторяющиеся даты"):
        parse_key_rate(xml)


@responses.activate
def test_get_key_rate_saves_and_normalizes(tmp_path):
    """Реальный запрос (моканый) сохраняет ответ и возвращает нормализованную таблицу."""
    responses.post(
        CBR_ENDPOINT,
        body=FIXTURE_OK.read_bytes(),
        status=200,
        content_type="text/xml",
    )

    with HttpClient() as http:
        client = CBRClient(http=http, raw_dir=tmp_path)
        result = client.get_key_rate(from_date=date(2014, 1, 1), to_date=date(2014, 1, 31))

    assert isinstance(result, KeyRateDownload)
    assert result.raw_path.exists()
    assert result.raw_path.with_name(result.raw_path.name + ".metadata.json").exists()
    assert result.unit == "percent_per_annum"
    assert result.rows == len(result.table)
    assert result.table["date"].iloc[-1] == date(2014, 1, 31)


@responses.activate
def test_get_key_rate_from_after_to_raises(tmp_path):
    """from_date позже to_date отклоняется без сетевого запроса."""
    with HttpClient() as http:
        client = CBRClient(http=http, raw_dir=tmp_path)
        with pytest.raises(ValueError, match="не может быть позже"):
            client.get_key_rate(from_date=date(2014, 2, 1), to_date=date(2014, 1, 1))
    assert len(responses.calls) == 0