import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def save_response(
    *,
    output_dir: Path,
    name: str,
    content: bytes,
    metadata: dict[str, Any],
) -> Path:
    """Сохранить исходный ответ и JSON-файл с метаданными."""
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("name должен содержать только имя файла без пути")

    metadata_path = output_dir / f"{name}.metadata.json"
    response_path = output_dir / name

    output_dir.mkdir(parents=True, exist_ok=True)

    if response_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Ответ или метаданные уже существуют: {response_path}")

    response_metadata = {
        **metadata,
        "file_name": name,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }

    metadata_content = json.dumps(
        response_metadata,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )

    with response_path.open("xb") as file:
        file.write(content)

    with metadata_path.open("x", encoding="utf-8") as file:
        file.write(metadata_content)
        file.write("\n")

    return response_path


_RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z_[a-f0-9]{8}$")


def run_dir(*, base_dir: Path, source: str) -> Path:
    """Создать каталог выгрузки для нового прогона с собственным идентификатором.

    Каталог имеет вид ``<base_dir>/<timestamp>_<run_id>/<source>``, где
    ``<run_id>`` — короткий уникальный идентификатор. Повторный запуск создания
    каталога с тем же идентификатором не допускается (защита от перезаписи).
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{uuid4().hex[:8]}"
    if not _RUN_ID_PATTERN.match(run_id):
        raise ValueError(f"Невалидный идентификатор прогона: {run_id}")

    output_dir = base_dir / run_id / source
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
