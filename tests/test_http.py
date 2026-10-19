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
