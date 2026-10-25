import pytest
import requests
import responses

from equity_factor_study.http import HttpClient


@responses.activate
def test_get_returns_response():
    url = "https://example.test/history"

    responses.get(
        url,
        json={"history": {"data": []}},
        status=200,
    )

    with HttpClient() as client:
        response = client.get(
            url,
            params={"start": 0},
        )

    assert response.status_code == 200
    assert response.json() == {"history": {"data": []}}
    assert responses.calls[0].request.url == f"{url}?start=0"


@responses.activate
def test_post_sends_xml():
    url = "https://example.test/soap"
    request_body = b"<request />"

    responses.post(
        url,
        body="<response />",
        status=200,
        content_type="text/xml",
    )

    with HttpClient() as client:
        response = client.post(
            url,
            data=request_body,
            headers={"Content-Type": "text/xml; charset=utf-8"},
        )

    assert response.content == b"<response />"
    assert responses.calls[0].request.body == request_body
    assert responses.calls[0].request.headers["Content-Type"] == "text/xml; charset=utf-8"


@responses.activate
def test_get_raises_http_error():
    url = "https://example.test/history"

    responses.get(url, status=403)

    with HttpClient() as client, pytest.raises(requests.HTTPError):
        client.get(url)

    assert len(responses.calls) == 1


@responses.activate
def test_get_propagates_timeout():
    url = "https://example.test/history"

    responses.get(
        url,
        body=requests.exceptions.ReadTimeout("Test timeout"),
    )

    with HttpClient(max_retries=0) as client, pytest.raises(requests.exceptions.ReadTimeout):
        client.get(url)


def test_client_configures_timeout():
    with HttpClient(
        connect_timeout=5,
        read_timeout=20,
    ) as client:
        assert client.timeout == (5, 20)


@responses.activate
def test_get_retries_on_connection_reset():
    """Разрыв соединения повторяется до успеха (этап 4, дросселирование)."""
    url = "https://example.test/history"

    responses.get(
        url,
        body=requests.exceptions.ChunkedEncodingError("Connection reset"),
    )

    with HttpClient(max_retries=1) as client, pytest.raises(requests.exceptions.ChunkedEncodingError):
        client.get(url)

    assert len(responses.calls) == 2  # первый + один повтор


@responses.activate
def test_get_does_not_retry_http_error():
    """Ошибка кода ответа (например, 403) не ретраится как сетевой сбой."""
    url = "https://example.test/history"
    responses.get(url, status=403)

    with HttpClient(max_retries=3) as client, pytest.raises(requests.HTTPError):
        client.get(url)

    assert len(responses.calls) == 1


def test_client_throttles_between_requests(monkeypatch):
    """Между запросами выдерживается минимальный интервал."""
    import time

    sleep_calls: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)

    with HttpClient(request_interval=0.5) as client:
        # Первый запрос не ждёт (нет предыдущего).
        client._throttle()
        assert sleep_calls == []
        # Второй — выдерживает интервал 0.5с.
        client._last_request_at = 100.0
        client._throttle()
        assert sleep_calls == [0.5]
