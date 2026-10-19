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
    ) -> None:
        self.timeout = (connect_timeout, read_timeout)
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

    def get(
        self,
        url: str,
        *,
        params: dict[str, str | int] | None = None,
    ) -> requests.Response:
        response = self.session.get(
            url,
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response

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
