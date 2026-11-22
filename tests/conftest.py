from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / f"eval-{uuid.uuid4().hex}.sqlite3"
    monkeypatch.setenv("EVALUATION_DB", str(db_path))
    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
