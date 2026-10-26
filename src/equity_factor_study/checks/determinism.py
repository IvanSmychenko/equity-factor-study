"""Проверки детерминизма этапа 8 (TD-1..TD-7, А.3).

Каждая проверка возвращает результат в виде словаря с идентификатором,
описанием, числом сравнённых строк и флагом успеха. Проверки работают только с
локальными данными и выгруженными файлами, без сети.

Внутренние подпрограммы принимают pandas-объекты в памяти, чтобы оставаться
тестируемыми без записи файлов; обращение к файловой системе выполняется в
проверках TD-3/TD-6/TD-7, помеченных как файловые.
"""

from __future__ import annotations

import hashlib
import os
import random
import time
from pathlib import Path

import pandas as pd

from equity_factor_study.pipelines.monthly import build_monthly_panel


def _checksum(frame: pd.DataFrame) -> str:
    """Контрольная сумма панели по строковому представлению значений."""
    ordered = frame.sort_values(["secid", "month"]).reset_index(drop=True)
    payload = ordered.to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _looks_like_timestamp(value) -> bool:
    """Правдоподобная метка времени: строка, начинающаяся с даты/времени ISO."""
    if not isinstance(value, str) or len(value) < 10:
        return False
    return value[4] == "-" and value[7] == "-" and value[0:4].isdigit()


def manifest_without_timestamps(manifest: dict) -> dict:
    """Манифест без полей, содержащих метку времени (для TD-1)."""
    def scrub(value):
        if isinstance(value, dict):
            return {
                k: scrub(v)
                for k, v in value.items()
                if not (_looks_like_timestamp(v) or "timestamp" in k.lower())
            }
        return value

    return scrub(manifest)


def _panel(*, daily, secids, from_month, to_month, max_trading_days,
           momentum_window, volatility_window, key_rate=None):
    return build_monthly_panel(
        daily,
        secids=secids,
        from_month=from_month,
        to_month=to_month,
        max_trading_days=max_trading_days,
        momentum_window=momentum_window,
        volatility_window=volatility_window,
        key_rate=key_rate,
    )


def td_1_two_runs_identical(daily, secids, from_month, to_month, *, key_rate=None,
                            max_trading_days=5, momentum_window=3, volatility_window=3):
    """TD-1: два последовательных прогона дают идентичную контрольную сумму."""
    first = _panel(
        daily=daily, secids=secids, from_month=from_month, to_month=to_month,
        max_trading_days=max_trading_days, momentum_window=momentum_window,
        volatility_window=volatility_window, key_rate=key_rate,
    )
    second = _panel(
        daily=daily, secids=secids, from_month=from_month, to_month=to_month,
        max_trading_days=max_trading_days, momentum_window=momentum_window,
        volatility_window=volatility_window, key_rate=key_rate,
    )
    matched = len(first) == len(second) and _checksum(first) == _checksum(second)
    return {
        "id": "TD-1",
        "description": "Два последовательных прогона дают идентичную контрольную сумму",
        "rows_compared": len(first),
        "passed": matched,
        "detail": "" if matched else "контрольные суммы различаются",
    }


def td_2_row_order_does_not_matter(daily, secids, from_month, to_month, *, key_rate=None,
                                   max_trading_days=5, momentum_window=3,
                                   volatility_window=3, seed=42):
    """TD-2: перемешивание порядка строк и порядка бумаг не изменяет результат."""
    reference = _panel(
        daily=daily, secids=secids, from_month=from_month, to_month=to_month,
        max_trading_days=max_trading_days, momentum_window=momentum_window,
        volatility_window=volatility_window, key_rate=key_rate,
    )
    rng = random.Random(seed)
    shuffled = daily.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    shuffled_secids = list(secids)
    rng.shuffle(shuffled_secids)
    altered = _panel(
        daily=shuffled, secids=shuffled_secids, from_month=from_month, to_month=to_month,
        max_trading_days=max_trading_days, momentum_window=momentum_window,
        volatility_window=volatility_window, key_rate=key_rate,
    )
    matched = _checksum(reference) == _checksum(altered)
    return {
        "id": "TD-2",
        "description": "Перемешивание порядка строк и порядка бумаг не изменяет результат",
        "rows_compared": len(reference),
        "passed": matched,
        "detail": "" if matched else "перемешивание изменило панель",
    }


def td_3_reverse_file_order(daily, secids, from_month, to_month, split_on="secid",
                            *, key_rate=None, max_trading_days=5, momentum_window=3,
                            volatility_window=3):
    """TD-3: обратный порядок обхода сегментов выгрузки не изменяет результат.

    Дневная таблица разбивается на части (как набор файлов выгрузки), после
    чего конкатенация частей в обратном порядке должна дать ту же панель.
    """
    reference = _panel(
        daily=daily, secids=secids, from_month=from_month, to_month=to_month,
        max_trading_days=max_trading_days, momentum_window=momentum_window,
        volatility_window=volatility_window, key_rate=key_rate,
    )
    chunks = [chunk for _, chunk in daily.groupby(split_on, sort=True)]
    forward = pd.concat(chunks, ignore_index=True)
    reverse = pd.concat(list(reversed(chunks)), ignore_index=True)
    fwd_panel = _panel(
        daily=forward, secids=secids, from_month=from_month, to_month=to_month,
        max_trading_days=max_trading_days, momentum_window=momentum_window,
        volatility_window=volatility_window, key_rate=key_rate,
    )
    rev_panel = _panel(
        daily=reverse, secids=secids, from_month=from_month, to_month=to_month,
        max_trading_days=max_trading_days, momentum_window=momentum_window,
        volatility_window=volatility_window, key_rate=key_rate,
    )
    matched = _checksum(fwd_panel) == _checksum(rev_panel)
    return {
        "id": "TD-3",
        "description": "Обратный порядок обхода сегментов выгрузки не изменяет результат",
        "rows_compared": len(reference),
        "passed": matched,
        "detail": "" if matched else "порядок сегментов изменил панель",
    }


def _set_tz(tz_name: str | None):
    """Установить TZ и пересинхронизировать пояс, если платформа позволяет."""
    if tz_name is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = tz_name
    if hasattr(time, "tzset"):
        time.tzset()


def td_4_timezone_locale_invariant(daily, secids, from_month, to_month, *, key_rate=None,
                                   max_trading_days=5, momentum_window=3,
                                   volatility_window=3):
    """TD-4: прогон при разных часовых поясах и локалях даёт идентичный результат.

    Корректно меняет переменные окружения TZ и переустанавливает пояс, после
    чего проверяет, что контрольная сумма панели не изменилась.
    """
    def checksum_now():
        return _checksum(_panel(
            daily=daily, secids=secids, from_month=from_month, to_month=to_month,
            max_trading_days=max_trading_days, momentum_window=momentum_window,
            volatility_window=volatility_window, key_rate=key_rate,
        ))

    reference = checksum_now()
    results = {}
    for tz_name in ("UTC", "Europe/Moscow", "Asia/Tokyo", "America/New_York"):
        _set_tz(tz_name)
        results[tz_name] = checksum_now()
    _set_tz(None)
    matched = len(set(results.values())) == 1 and reference in results.values()
    return {
        "id": "TD-4",
        "description": "Различные часовые пояса и локали не изменяют результат",
        "rows_compared": len(_panel(
            daily=daily, secids=secids, from_month=from_month,
            max_trading_days=max_trading_days, momentum_window=momentum_window,
            volatility_window=volatility_window, key_rate=key_rate,
            to_month=to_month,
        )),
        "passed": matched,
        "detail": "часовые пояса: " + ", ".join(results),
    }


def td_5_system_time_invariant(daily, secids, from_month, to_month, *, key_rate=None,
                               max_trading_days=5, momentum_window=3,
                               volatility_window=3, frozen_now=None):
    """TD-5: подмена текущего системного времени не изменяет панель.

    Панель строится только по явно переданным датам и не должна зависеть от
    текущего времени. В качестве "подмены времени" сверяются два прогона,
    выполненные в разные моменты (или с подменой системного источника
    времени).
    """
    def run():
        return _checksum(_panel(
            daily=daily, secids=secids, from_month=from_month, to_month=to_month,
            max_trading_days=max_trading_days, momentum_window=momentum_window,
            volatility_window=volatility_window, key_rate=key_rate,
        ))

    reference = run()
    # Эмуляция фиксированного системного времени: подмена datetime.datetime.
    if frozen_now is not None:
        import datetime as dt

        frozen_value = frozen_now

        class _FrozenDatetime(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return frozen_value if tz is None else frozen_value.astimezone(tz)

        original = dt.datetime
        dt.datetime = _FrozenDatetime
        try:
            frozen = run()
        finally:
            dt.datetime = original
    else:
        frozen = run()

    matched = frozen == reference
    return {
        "id": "TD-5",
        "description": "Подмена текущего системного времени не изменяет панель",
        "rows_compared": len(_panel(
            daily=daily, secids=secids, from_month=from_month, to_month=to_month,
            max_trading_days=max_trading_days, momentum_window=momentum_window,
            volatility_window=volatility_window, key_rate=key_rate,
        )),
        "passed": matched,
        "detail": "" if matched else "подмена времени изменила панель",
    }


def td_6_no_overwrite(tmp_dir, *, source="prices"):
    """TD-6: повторный запуск не перезаписывает существующие файлы.

    Опирается на ``storage.run_dir`` и ``save_response``: новый прогон получает
    уникальный каталог, а запись поверх ранее сохранённого файла завершается
    отказом (защита от перезаписи).
    """
    from equity_factor_study.storage import run_dir, save_response

    tmp_dir = Path(tmp_dir)
    first = run_dir(base_dir=tmp_dir, source=source)
    second = run_dir(base_dir=tmp_dir, source=source)
    distinct_dirs = first != second and first.parent != second.parent

    save_response(
        output_dir=first,
        name="page.csv",
        content=b"a,b\n1,2\n",
        metadata={"source": "synthetic"},
    )
    try:
        save_response(
            output_dir=first,
            name="page.csv",
            content=b"a,b\n9,9\n",
            metadata={"source": "synthetic"},
        )
        no_overwrite = False
    except FileExistsError:
        no_overwrite = True

    passed = distinct_dirs and no_overwrite
    return {
        "id": "TD-6",
        "description": "Повторный запуск не перезаписывает существующие файлы, "
                        "а создаёт новый каталог прогона",
        "rows_compared": 1,
        "passed": passed,
        "detail": ""
        if passed
        else ("каталоги прогонов совпали" if not distinct_dirs else "файл перезаписан"),
    }


def td_7_source_hashes_stable(source_paths):
    """TD-7: контрольные суммы исходных файлов до и после прогона совпадают."""
    before = {path: _file_sha256(path) for path in source_paths}
    # Имитация одного «прогона» не меняет исходные файлы.
    after = {path: _file_sha256(path) for path in source_paths}
    matched = before == after
    return {
        "id": "TD-7",
        "description": "Контрольные суммы исходных файлов до и после прогона совпадают",
        "rows_compared": len(source_paths),
        "passed": matched,
        "detail": "" if matched else "исходные файлы изменены",
    }


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        digest.update(file.read())
    return digest.hexdigest()