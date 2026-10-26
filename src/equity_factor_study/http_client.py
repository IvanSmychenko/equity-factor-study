import time
from typing import Self

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class HttpClient:
    def __init__(
        self,
        *,
        connect_timeout: float = 15,
        read_timeout: float = 90,
        max_retries: int = 3,
        request_interval: float = 0.0,
    ) -> None:
        self.timeout = (connect_timeout, read_timeout)
        # Минимальный интервал между запросами (сек.) — дросселирование,
        # чтобы не перегружать источник при массовом сборе.
        self.request_interval = max(0.0, request_interval)
        self._last_request_at: float | None = None
        self.max_retries = max_retries
        self.session = requests.Session()

        self.session.headers.update(
            {
                "User-Agent": "equity-factor-study/0.1.0",
            }
        )

        retry_policy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
            raise_on_status=False,
        )

        adapter = HTTPAdapter(max_retries=retry_policy)

        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _throttle(self) -> None:
        """Выдержать минимальный интервал между запросами."""
        if self.request_interval <= 0 or self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)

    def get(
        self,
        url: str,
        *,
        params: dict[str, str | int] | None = None,
    ) -> requests.Response:
        # Ошибки сетевого уровня, при которых имеет смысл повторить запрос.
        # Таймауты не ретраим здесь (их обрабатывает политика Retry выше),
        # а ошибки кода ответа (HTTPError) уходят через raise_for_status после
        # успешного сетевого вызова.
        retryable = (
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                self._throttle()
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                )
                self._last_request_at = time.monotonic()
                response.raise_for_status()
                return response
            except retryable as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                time.sleep(2**attempt)  # прогрессивный backoff

        raise last_error if last_error else requests.exceptions.ConnectionError("no response")

    def post(
        self,
        url: str,
        *,
        data: bytes,
        headers: dict[str, str] | None = None,
    ) -> requests.Response:
        response = self.session.post(
            url,
            data=data,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
