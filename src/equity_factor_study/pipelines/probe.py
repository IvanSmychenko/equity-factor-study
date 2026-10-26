from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

from equity_factor_study.clients.moex import MOEXClient
from equity_factor_study.http_client import HttpClient


def main() -> None:
    print("Запуск пробного запроса к MOEX", flush=True)
    print(f"Рабочая папка: {Path.cwd()}", flush=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{uuid4().hex[:8]}"
    raw_dir = Path("data") / "raw" / run_id / "moex"

    print(f"Каталог выгрузки: {raw_dir.resolve()}", flush=True)
    print("Запрашиваем историю SBER за 1–7 сентября 2026 года...", flush=True)

    with HttpClient() as http:
        client = MOEXClient(
            http=http,
            raw_dir=raw_dir,
        )

        page = client.get_history_page(
            security="SBER",
            from_date=date(2026, 9, 1),
            till_date=date(2026, 9, 7),
            board="TQBR",
            start=0,
        )

    print(f"\nИсходный ответ сохранён: {page.raw_path.resolve()}")
    print(f"Получено строк на странице: {len(page.table)}")

    if page.table.empty:
        print("Таблица history пустая. Требуется проверить исходный ответ.")
    else:
        print("\nПервые строки:")
        print(page.table.head().to_string(index=False))

    if not page.cursor.empty:
        print("\nКурсор:")
        print(page.cursor.to_string(index=False))

    print("\nПробный запрос и разбор ответа завершены.")


if __name__ == "__main__":
    main()