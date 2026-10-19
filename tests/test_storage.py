import hashlib
import json

import pytest

from equity_factor_study.storage import save_response


def test_save_response_creates_files(tmp_path):
    output_dir = tmp_path / "raw" / "moex"
    content = b'{"history": {"columns": [], "data": []}}'
    metadata = {
        "source": "moex",
        "request_method": "GET",
        "retrieved_at_utc": "2026-10-18T19:48:00+00:00",
        "status_code": 200,
    }

    response_path = save_response(
        output_dir=output_dir,
        name="history.json",
        content=content,
        metadata=metadata,
    )

    metadata_path = output_dir / "history.json.metadata.json"
    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert response_path == output_dir / "history.json"
    assert response_path.read_bytes() == content
    assert saved_metadata == {
        **metadata,
        "file_name": "history.json",
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }

    assert "sha256" not in metadata


@pytest.mark.parametrize(
    "existing_file",
    ["history.json", "history.json.metadata.json"],
)
def test_save_response_does_not_overwrite_existing_files(
    tmp_path,
    existing_file,
):
    existing_path = tmp_path / existing_file
    existing_path.write_bytes(b"original")

    with pytest.raises(FileExistsError):
        save_response(
            output_dir=tmp_path,
            name="history.json",
            content=b"new content",
            metadata={},
        )

    assert existing_path.read_bytes() == b"original"


@pytest.mark.parametrize(
    "name",
    ["", ".", "..", "../history.json", "nested/history.json", r"nested\history.json"],
)
def test_save_response_rejects_invalid_name(tmp_path, name):
    with pytest.raises(ValueError):
        save_response(
            output_dir=tmp_path,
            name=name,
            content=b"response",
            metadata={},
        )

    assert list(tmp_path.iterdir()) == []


def test_invalid_metadata_does_not_create_response_file(tmp_path):
    with pytest.raises(TypeError):
        save_response(
            output_dir=tmp_path,
            name="history.json",
            content=b"response",
            metadata={"unsupported_value": object()},
        )

    assert list(tmp_path.iterdir()) == []
