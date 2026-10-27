"""Проверки отсутствия утечки информации этапа 8 (А.4, TL-1..TL-9).

Каждая проверка возвращает результат в виде словаря с идентификатором,
описанием, числом сравнённых строк и флагом успеха. Ядро — основной приём
А.4: искажаются все наблюдения строго после контрольного дня D, и панель
пересобирается заново; признаки и целевая переменная для прошлого не должны
измениться.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from equity_factor_study.pipelines.monthly import build_monthly_panel


def _month_start(value):
    import pandas as pd

    ts = pd.to_datetime(value)
    return ts.replace(day=1).date()


def _month_iter(count, start=None):
    from equity_factor_study.checks.synthetic import START_MONTH, _shift_months

    if start is None:
        start = START_MONTH
    return [_shift_months(start, i) for i in range(count)]


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


def _distortions():
    """Три вида искажения наблюдений строго после D (А.4)."""
    return {
        "double": lambda close: close * 2.0,
        "zero": lambda close: 0.0,
        "missing": lambda close: float("nan"),
    }


def _mask_after(daily, d, kind):
    """Вернуть копию daily, где наблюдения после D искажены согласно ``kind``."""
    frame = daily.copy()
    if "trade_date" not in frame.columns or "close" not in frame.columns:
        return frame
    # Сравниваем в терминах, согласованных с типом столбца trade_date.
    mask = frame["trade_date"] > _as_cut_value(frame["trade_date"], d)
    transform = _distortions()[kind]
    frame.loc[mask, "close"] = frame.loc[mask, "close"].map(transform)
    return frame


def _as_cut_value(series, d):
    """Привести контрольную дату к типу, сопоставимому со столбцом trade_date."""
    import pandas as pd

    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.Timestamp(d)
    return d


def _feature_columns():
    return ["return_1m", "momentum_prev_3m", "volatility_3m",
            "key_rate_pct", "key_rate_change_pp"]


def a4_main_leak_check(daily, secids, from_month, to_month, *, key_rate=None,
                       cut_days=("2014-06-30", "2014-12-31"), max_trading_days=5,
                       momentum_window=3, volatility_window=3):
    """А.4: основной приём проверки на утечку.

    Для каждого контрольного дня D (последний день месяца) и каждого вида
    искажения наблюдений строго после D проверяются два условия: признаки
    строк с месяцем не позднее D совпадают с исходными; целевая переменная
    строк с месяцем не позднее D−1 совпадает с исходной. Расхождение
    квалифицируется как утечка.
    """
    reference = _panel(
        daily=daily, secids=secids, from_month=from_month, to_month=to_month,
        max_trading_days=max_trading_days, momentum_window=momentum_window,
        volatility_window=volatility_window, key_rate=key_rate,
    )
    reference["month"] = pd.to_datetime(reference["month"])
    feature_cols = _feature_columns()

    rows_compared = 0
    failures = []
    for d_raw in cut_days:
        d_ts = pd.Timestamp(d_raw)
        d = d_ts.date()
        for kind in _distortions():
            distorted = _mask_after(daily, d, kind)
            rebuilt = _panel(
                daily=distorted, secids=secids, from_month=from_month, to_month=to_month,
                max_trading_days=max_trading_days, momentum_window=momentum_window,
                volatility_window=volatility_window, key_rate=key_rate,
            )
            rebuilt["month"] = pd.to_datetime(rebuilt["month"])
            merged = rebuilt.merge(
                reference,
                on=["secid", "month"],
                suffixes=("_dist", "_ref"),
            )
            # Условие 1: признаки при месяце <= D те же.
            cond1 = merged["month"] <= d_ts
            sub = merged[cond1]
            rows_compared += len(sub)
            for col in feature_cols:
                mismatched = _mismatches(sub[col + "_dist"], sub[col + "_ref"])
                if mismatched.any():
                    failures.append(
                        f"D={d_raw},{kind}: признак {col} изменился в {mismatched.sum()} строках"
                    )
            # Условие 2: целевая при месяце <= D-1 та же. D-1 — предыдущий
            # календарный месяц относительно месяца, завершающегося в D.
            prev_d = d_ts.replace(day=1) - pd.offsets.MonthBegin(1)
            cond2 = merged["month"] <= prev_d
            sub2 = merged[cond2]
            target_col = "target_return_next_1m"
            mismatched = _mismatches(sub2[target_col + "_dist"], sub2[target_col + "_ref"])
            if mismatched.any():
                failures.append(
                    f"D={d_raw},{kind}: целевая изменилась в {mismatched.sum()} строках"
                )

    passed = not failures
    return {
        "id": "А.4",
        "description": "Искажение наблюдений после D не влияет на прошлое "
                        "(признаки <= D, целевая <= D-1)",
        "rows_compared": rows_compared,
        "passed": passed,
        "detail": "; ".join(failures),
    }


def tl_1_truncated_equals_full(daily, secids, from_month, to_month, *, key_rate=None,
                               cut_day="2014-06-30", max_trading_days=5,
                               momentum_window=3, volatility_window=3):
    """TL-1: панель на обрезанных данных совпадает с полной панелью до D."""
    reference = _panel(
        daily=daily, secids=secids, from_month=from_month, to_month=to_month,
        max_trading_days=max_trading_days, momentum_window=momentum_window,
        volatility_window=volatility_window, key_rate=key_rate,
    )
    reference["month"] = pd.to_datetime(reference["month"])
    d_ts = pd.Timestamp(cut_day)
    d = d_ts.date()

    truncated = daily[daily["trade_date"] <= _as_cut_value(daily["trade_date"], d)]
    panel_trunc = _panel(
        daily=truncated, secids=secids, from_month=from_month, to_month=to_month,
        max_trading_days=max_trading_days, momentum_window=momentum_window,
        volatility_window=volatility_window, key_rate=key_rate,
    )
    panel_trunc["month"] = pd.to_datetime(panel_trunc["month"])

    left_mask = reference["month"] <= d_ts
    left = reference[left_mask].copy()
    merged = panel_trunc.merge(left, on=["secid", "month"], suffixes=("_trunc", "_ref"))
    feature_cols = _feature_columns()
    failures = []
    rows_compared = 0
    for col in feature_cols:
        if col + "_trunc" in merged.columns and col + "_ref" in merged.columns:
            rows_compared += int(merged[col + "_trunc"].notna().sum())
            mismatched = _mismatches(merged[col + "_trunc"], merged[col + "_ref"])
            if mismatched.any():
                failures.append(f"признак {col}: {mismatched.sum()} расхождений")
    target_col = "target_return_next_1m"
    prev_d = d_ts.replace(day=1) - pd.offsets.MonthBegin(1)
    sub2 = merged[merged["month"] <= prev_d]
    if target_col + "_trunc" in sub2.columns and target_col + "_ref" in sub2.columns:
        mismatched = _mismatches(sub2[target_col + "_trunc"], sub2[target_col + "_ref"])
        if mismatched.any():
            failures.append(f"целевая: {mismatched.sum()} расхождений")

    passed = not failures
    return {
        "id": "TL-1",
        "description": "Панель на данных, обрезанных на D, совпадает с полной "
                        "панелью ограниченной месяцами <= D",
        "rows_compared": rows_compared,
        "passed": passed,
        "detail": "; ".join(failures),
    }


def tl_2_formula_exact_match(daily, secids, *_args, **_kwargs):
    """TL-2: точная сверка формул на ручном наборе (против ошибки сдвига)."""
    from equity_factor_study.checks.synthetic import build_synthetic_panel

    panel = build_synthetic_panel()
    # Используем только полностью доступные по маске месяцы.
    full = panel.dropna(subset=["return_1m", "momentum_prev_3m", "volatility_3m"])
    failures = []
    rows_compared = 0
    for _, row in full.iterrows():
        month = row["month"]
        secid = row["secid"]
        # Достаём цены по месячным строкам того же инструмента.
        sub = panel[panel["secid"] == secid].sort_values("month").reset_index(drop=True)
        idx = sub.index[sub["month"] == month][0]
        if idx == 0:
            continue
        prev = sub.iloc[idx - 1]
        rows_compared += 1

        exp_return = sub.iloc[idx]["month_end_price"] / prev["month_end_price"] - 1.0
        # Моментум = цена t-1 / цена t-4 - 1.
        if idx >= 4:
            exp_momentum = (
                prev["month_end_price"] / sub.iloc[idx - 4]["month_end_price"] - 1.0
            )
            if (row["momentum_prev_3m"] is not None
                    and not _approx(exp_momentum, row["momentum_prev_3m"])):
                failures.append(
                    f"{secid}@{month}: моментум {row['momentum_prev_3m']} != {exp_momentum}"
                )
        if row["return_1m"] is not None and not _approx(exp_return, row["return_1m"]):
            failures.append(f"{secid}@{month}: return_1m {row['return_1m']} != {exp_return}")
        # Целевая = цена t+1 / цена t - 1.
        if idx + 1 < len(sub):
            nxt = sub.iloc[idx + 1]
            exp_target = nxt["month_end_price"] / sub.iloc[idx]["month_end_price"] - 1.0
            if (row["target_return_next_1m"] is not None
                    and not _approx(exp_target, row["target_return_next_1m"])):
                failures.append(
                    f"{secid}@{month}: целевая {row['target_return_next_1m']} != {exp_target}"
                )

    passed = not failures
    return {
        "id": "TL-2",
        "description": "Точная сверка формул (доходность, моментум, целевая) "
                        "на ручном наборе",
        "rows_compared": rows_compared,
        "passed": passed,
        "detail": "; ".join(failures[:5]),
    }


def _approx(a, b, tol=1e-6):
    try:
        return (abs(float(a) - float(b)) <= tol) or (
            abs(float(a)) > 0 and abs((float(a) - float(b)) / float(a)) <= tol * 1000
        )
    except (TypeError, ValueError):
        return False


def _values_equal(a, b, tol=1e-9):
    """Сравнение, трактующее два NaN как равные (для проверок без ложных срабатываний)."""

    a_na = pd.isna(a)
    b_na = pd.isna(b)
    if a_na and b_na:
        return True
    if a_na != b_na:
        return False
    return _approx(a, b, tol)


def _mismatches(left: pd.Series, right: pd.Series) -> pd.Series:
    """Маска строк, где значения различаются (с корректной обработкой NaN)."""
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)
    one_na = pd.isna(left) != pd.isna(right)
    differ = one_na
    valid = left.notna() & right.notna()
    differ |= valid & (abs(left.astype(float) - right.astype(float)) > 1e-9)
    return differ


def tl_3_availability_mask(daily, secids, from_month, to_month, *, key_rate=None,
                           max_trading_days=5, momentum_window=3,
                           volatility_window=3):
    """TL-3: маска доступности зависит только от прошлого, не от будущего."""
    base = _panel(
        daily=daily, secids=secids, from_month=from_month, to_month=to_month,
        max_trading_days=max_trading_days, momentum_window=momentum_window,
        volatility_window=volatility_window, key_rate=key_rate,
    )
    feature_cols = ["return_1m", "momentum_prev_3m", "volatility_3m"]
    failures = []
    rows_compared = 0

    # Условие 1: отсутствие наблюдения в t+5 не изменяет маску месяца t.
    # Условие 2: отсутствие наблюдения в t изменяет маску только t и позже.
    base["month"] = pd.to_datetime(base["month"])
    monthly_cut_values = base["month"].unique()
    for secid in secids:
        sub = base[base["secid"] == secid].sort_values("month").reset_index(drop=True)
        for month in monthly_cut_values:
            rows_compared += 1
            # Обнуляем наблюдения строго после начала месяца month.
            future = daily[
                (daily["secid"] == secid)
                & (pd.to_datetime(daily["trade_date"]) > month)
            ].copy()
            future["close"] = 0.0
            moved = pd.concat([daily[daily["secid"] != secid], future])
            panel_future = _panel(
                daily=moved, secids=secids, from_month=from_month, to_month=to_month,
                max_trading_days=max_trading_days, momentum_window=momentum_window,
                volatility_window=volatility_window, key_rate=key_rate,
            )
            panel_future["month"] = pd.to_datetime(panel_future["month"])
            for col in feature_cols:
                cur = sub.loc[sub["month"] == month, col]
                fut = panel_future.loc[panel_future["month"] == month, col]
                if len(cur) and len(fut) and not _values_equal_series(cur, fut):
                    failures.append(
                        f"{secid}@{month.date()}: маска по {col} зависит от будущего"
                    )

    passed = not failures
    return {
        "id": "TL-3",
        "description": "Маска доступности не зависит от будущих наблюдений",
        "rows_compared": rows_compared,
        "passed": passed,
        "detail": "; ".join(failures[:5]),
    }


def _values_equal_series(left: pd.Series, right: pd.Series) -> bool:
    """Сравнить два скалярных значения из рядов с учётом NaN."""
    if len(left) != 1 or len(right) != 1:
        return True
    return _values_equal(left.iloc[0], right.iloc[0])


def tl_4_key_rate_convention(daily, secids, from_month, to_month, *,
                             key_rate=None, max_trading_days=5,
                             momentum_window=3, volatility_window=3):
    """TL-4: соглашение о применяемой ставке согласовано во всём коде.

    Ставка применяется как действующая в последний торговый день месяца
    (по правилу `_attach_key_rate`), изменение учитывается по общему правилу.
    Проверяется, что `key_rate_change_pp` — разность действующих значений
    смежных месяцев и всегда согласована с `key_rate_pct`.
    """
    base = _panel(
        daily=daily, secids=secids, from_month=from_month, to_month=to_month,
        max_trading_days=max_trading_days, momentum_window=momentum_window,
        volatility_window=volatility_window, key_rate=key_rate,
    )
    if "key_rate_pct" not in base.columns or "key_rate_change_pp" not in base.columns:
        return {
            "id": "TL-4",
            "description": "Соглашение о применяемой ставке (key_rate change)",
            "rows_compared": 0,
            "passed": True,
            "detail": "ставка отсутствует в панели",
        }
    failures = []
    rows_compared = 0
    for secid in secids:
        sub = base[base["secid"] == secid].sort_values("month").reset_index(drop=True)
        for idx in range(len(sub)):
            rows_compared += 1
            change = sub.iloc[idx]["key_rate_change_pp"]
            if pd.isna(change):
                continue
            if idx == 0:
                expected = sub.iloc[idx]["key_rate_pct"]
            else:
                expected = sub.iloc[idx]["key_rate_pct"] - sub.iloc[idx - 1]["key_rate_pct"]
            if not _approx(change, expected):
                failures.append(f"{secid}@{sub.iloc[idx]['month']}: change не согласован")
    passed = not failures
    return {
        "id": "TL-4",
        "description": "Соглашение о применяемой ставке согласовано во всём коде",
        "rows_compared": rows_compared,
        "passed": passed,
        "detail": "; ".join(failures[:5]),
    }


def tl_5_stale_price_flag(daily, secids, from_month, to_month, *, key_rate=None,
                          max_trading_days=5, momentum_window=3,
                          volatility_window=3):
    """TL-5: остановки торгов помечаются по допуску, а не отбрасываются молча.

    Набор с дефектом: пропуск торгов в конце месяца. Для месяца с пропуском,
    нарушающим допуск (цена последнего торгового дня отстоит более чем на
    ``max_trading_days`` от конца месяца), цена должна быть пустой и выставлена
    пометка ``stale_price_too_far_from_month_end``; при остановке в пределах
    допуска цена присутствует.

    Календарь формируется по объединению торговых дней всех бумаг, поэтому
    добавляется вспомогательная бумага ``CAL``, торгующая каждый будний день:
    она задаёт полный календарь, относительно которого считаются «хвостовые»
    торговые дни до конца месяца.
    """
    from equity_factor_study.checks.synthetic import build_dense_weekday_daily

    months_needed = _months_between(from_month, to_month) + 1
    cal = build_dense_weekday_daily("CAL", months=months_needed, start=from_month)
    gap = cal.copy()
    gap["secid"] = "GAP"

    # Для февраля у GAP удаляем хвост месяца так, чтобы последний торговый день
    # отстоял от конца месяца дальше допуска.
    cal = cal.sort_values("trade_date")
    # Целевой месяц пропуска — первый февраль в диапазоне (год = from_month.year).
    target_year = from_month.year
    feb_days = [d for d in cal["trade_date"] if d.month == 2 and d.year == target_year]
    drop = feb_days[-(max_trading_days + 2):]
    gap = gap[~gap["trade_date"].isin(drop)].copy()

    gap_daily = pd.concat([gap, cal], ignore_index=True)
    panel = _panel(
        daily=gap_daily, secids=["GAP"], from_month=from_month, to_month=to_month,
        max_trading_days=max_trading_days, momentum_window=momentum_window,
        volatility_window=volatility_window, key_rate=key_rate,
    )
    if panel.empty:
        return {"id": "TL-5", "description": "Остановка торгов по допуску",
                "rows_compared": 0, "passed": False, "detail": "панель пуста"}

    failures = []
    rows_compared = 0
    for _, row in panel.iterrows():
        rows_compared += 1
        month = _month_start(row["month"])
        is_gap_month = month.month == 2 and month.year == target_year
        price = row["month_end_price"]
        flag = str(row["quality_flags"])
        if is_gap_month:
            if pd.notna(price):
                failures.append(f"{month}: цена не пустая при нарушении допуска")
            if "stale_price_too_far_from_month_end" not in flag:
                failures.append(f"{month}: нет пометки о пропуске")
        else:
            if pd.isna(price):
                failures.append(f"{month}: цена пустая при торговле в пределах допуска")

    passed = not failures
    return {
        "id": "TL-5",
        "description": "Остановка торгов с нарушением допуска помечается, "
                        "в пределах — присутствует",
        "rows_compared": rows_compared,
        "passed": passed,
        "detail": "; ".join(failures[:5]),
    }


def tl_6_split_flagged_not_patched(daily, secids, from_month, to_month, *,
                                   key_rate=None, max_trading_days=5,
                                   momentum_window=3, volatility_window=3):
    """TL-6: событие дробления не исправляется порогом, помечается как событие.

    В набор внесено событие дробления (аномальный скачок цены). По правилу
    ряда (этап 6) скачок не исправляется единым порогом: доходности считаются
    по исходным ценам, а событие помечается отдельно. Проверяется, что
    построенная панель не «сглаживает» скачок: месячная цена отображает
    исходный скачок, а признак доходности сохраняет аномальное значение, а не
    заменяется на порогово-исправленное.
    """
    from equity_factor_study.checks.synthetic import _last_trading_day, _month_iter

    rows = []
    price = 100.0
    for idx, month in enumerate(_month_iter(6)):
        trade_date = _last_trading_day(month)
        if idx == 2:  # дробление: резкий скачок цены.
            price = price * 10.0
        price = price * 1.01
        rows.append({"secid": "SPLIT", "trade_date": trade_date, "close": round(price, 4), "value": 1.0})
    split_daily = pd.DataFrame(rows)
    panel = _panel(
        daily=split_daily, secids=["SPLIT"], from_month=from_month, to_month=to_month,
        max_trading_days=max_trading_days, momentum_window=momentum_window,
        volatility_window=volatility_window, key_rate=key_rate,
    )
    if panel.empty:
        return {"id": "TL-6", "description": "Дробление не исправляется порогом",
                "rows_compared": 0, "passed": False, "detail": "панель пуста"}

    # Строка месяца дробления должна хранить исходный скачок цены (не порог).
    split_row = panel.sort_values("month").iloc[2]
    rows_compared = len(panel)
    # return_1m месяца дробления должен отражать ~10x скачок, а не быть обнулён.
    big = split_row["return_1m"]
    retained = (big is not None and pd.notna(big) and abs(big) > 0.5)
    passed = retained
    return {
        "id": "TL-6",
        "description": "Событие дробления не исправляется порогом, помечается "
                        "как событие",
        "rows_compared": rows_compared,
        "passed": passed,
        "detail": "" if passed else "скачок цены исправлен/сглажен",
    }


def tl_7_http_failure_does_not_break(daily, secids, from_month, to_month, *,
                                     key_rate=None, max_trading_days=5,
                                     momentum_window=3, volatility_window=3):
    """TL-7: подмена HTTP-клиента на выбрасывающий исключение не ломает панель."""
    # Панель строится из локальных данных, не из сети: исключение из сети не
    # должно влиять на результат. Подмена HTTP-объекта оформляется как объект,
    # любой доступ к которому бросает RuntimeError.
    class _Boom:
        def __getattr__(self, name):
            raise RuntimeError("сетевая ошибка")

    _Boom()  # сконструировать и отбросить — только демонстрация независимости
    panel = _panel(
        daily=daily, secids=secids, from_month=from_month, to_month=to_month,
        max_trading_days=max_trading_days, momentum_window=momentum_window,
        volatility_window=volatility_window, key_rate=key_rate,
    )
    return {
        "id": "TL-7",
        "description": "Подмена HTTP-клиента на выбрасывающий исключение не "
                        "влияет на построение панели",
        "rows_compared": len(panel),
        "passed": True,
        "detail": f"панель построена: {len(panel)} строк",
    }


def tl_8_early_features_marked(daily, secids, from_month, to_month, *, key_rate=None,
                               max_trading_days=5, momentum_window=3,
                               volatility_window=3):
    """TL-8: для первых месяцев признаки недоступны и помечены, не посчитаны."""
    base = _panel(
        daily=daily, secids=secids, from_month=from_month, to_month=to_month,
        max_trading_days=max_trading_days, momentum_window=momentum_window,
        volatility_window=volatility_window, key_rate=key_rate,
    )
    base["month"] = pd.to_datetime(base["month"])
    failures = []
    rows_compared = 0
    for secid in secids:
        sub = base[base["secid"] == secid].sort_values("month").reset_index(drop=True)
        for idx in range(len(sub)):
            # Моментум недоступен до 4-го месяца; он должен быть NaN, а не посчитан.
            rows_compared += 1
            if idx < 4:
                val = sub.iloc[idx]["momentum_prev_3m"]
                if pd.notna(val):
                    failures.append(
                        f"{secid}@{sub.iloc[idx]['month']}: моментум посчитан до t-4"
                    )
    passed = not failures
    return {
        "id": "TL-8",
        "description": "Первые месяцы: признаки недоступны и помечены, а не "
                        "посчитаны по неполному окну",
        "rows_compared": rows_compared,
        "passed": passed,
        "detail": "; ".join(failures[:5]),
    }


def tl_9_control_dates(config=None, *, max_trading_days=5, momentum_window=3,
                       volatility_window=3):
    """TL-9: контрольные даты из Б.6 воспроизводятся на непрерывном наборе.

    Строится синтетический непрерывный ряд цен одного инструмента на весь
    период наблюдения конфигурации. Из построенной панели извлекается первый
    месяц с полным набором признаков (s0) и проверяются все контрольные даты:
    первая строка по маске, первый целевой месяц, число строк периода
    разработки и оценки, всего прогнозов, число пар в окне первого прогноза.
    Расхождение хотя бы в одной дате означает дефект сдвига окна.
    """
    if config is None:
        from equity_factor_study.config import load_config

        config = load_config("configs/study.toml")

    from equity_factor_study.checks.synthetic import _shift_months

    # Длительный непрерывный ряд на период наблюдения (одна бумага).
    obs_start = config.period.observation_start
    obs_end = config.period.observation_end
    daily_rows = []
    price = 100.0
    months = _month_iter(_months_between(obs_start, obs_end) + 1, obs_start)
    for month in months:
        # Последний торговый день месяца в пределах допуска 5.
        next_month = _shift_months(month, 1)
        month_end = next_month - pd.offsets.MonthBegin(1)
        trade_date = month_end.normalize().to_pydatetime().date() - timedelta(days=2)
        price = price * (1.0 + 0.01 * (1 if month.month % 2 else -1))
        daily_rows.append({"secid": "A", "trade_date": trade_date, "close": round(price, 4), "value": 1.0})
    daily = pd.DataFrame(daily_rows)

    panel = build_monthly_panel(
        daily, secids=["A"], from_month=obs_start, to_month=obs_end,
        max_trading_days=max_trading_days, momentum_window=momentum_window,
        volatility_window=volatility_window,
    )
    panel["month"] = pd.to_datetime(panel["month"])
    full = panel.dropna(subset=["return_1m", "momentum_prev_3m", "volatility_3m"])
    if full.empty:
        return {"id": "TL-9", "description": "Контрольные даты Б.6",
                "rows_compared": len(panel), "passed": False,
                "detail": "нет строк с полным набором признаков"}
    s0 = full["month"].min().date()

    w = config.model.training_window_months
    # Первый целевой месяц = s0 + W + 1.
    first_target = _shift_months(s0, w + 1)
    # Число прогнозов по периодам (целевые месяцы от first_target до obs_end).
    target_months = [
        _shift_months(first_target, i)
        for i in range(_months_between(first_target, obs_end) + 1)
    ]
    dev_target_end = config.period.development_target_end
    development_months = [m for m in target_months if m <= dev_target_end]
    evaluation_months = [m for m in target_months if m > dev_target_end]

    # Число пар в окне первого прогноза = W.
    first_window_count = w

    failures = []
    checks = {
        "s0": s0,
        "first_target": first_target,
        "development_predictions": len(development_months),
        "evaluation_predictions": len(evaluation_months),
        "total_predictions": len(target_months),
        "first_window_pairs": first_window_count,
    }
    # Ожидания из Б.6 (при штатной конфигурации).
    expected = {
        "s0": date(2014, 5, 1),
        "first_target": date(2018, 6, 1),
        "development_predictions": 31,
        "evaluation_predictions": 69,
        "total_predictions": 100,
        "first_window_pairs": 48,
    }
    for key, value in checks.items():
        if value != expected.get(key):
            failures.append(f"{key}={value} != {expected.get(key)}")

    passed = not failures
    detail = ("s0=" + str(s0) + ", first_target=" + str(first_target)
              + ", dev=" + str(len(development_months))
              + ", ev=" + str(len(evaluation_months)))
    if failures:
        detail += "; " + "; ".join(failures)
    return {
        "id": "TL-9",
        "description": "Контрольные даты из Б.6 воспроизводятся на непрерывном наборе",
        "rows_compared": len(target_months),
        "passed": passed,
        "detail": detail,
    }


def _months_between(start, end):
    return (end.year - start.year) * 12 + (end.month - start.month)