"""SQLite storage layer with explicit WAL transactions."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    batch_id          TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    current_version   INTEGER NOT NULL,
    version_created_at TEXT NOT NULL,
    photo_count       INTEGER NOT NULL,
    group_count       INTEGER NOT NULL,
    evaluable         INTEGER NOT NULL,
    overall_accuracy  REAL,
    reason            TEXT
);

CREATE TABLE IF NOT EXISTS photos (
    batch_id    TEXT NOT NULL,
    photo_id    TEXT NOT NULL,
    species     TEXT NOT NULL,
    features    TEXT NOT NULL,
    PRIMARY KEY (batch_id, photo_id),
    FOREIGN KEY (batch_id) REFERENCES batches(batch_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS edges (
    batch_id  TEXT NOT NULL,
    photo_a   TEXT NOT NULL,
    photo_b   TEXT NOT NULL,
    PRIMARY KEY (batch_id, photo_a, photo_b),
    FOREIGN KEY (batch_id) REFERENCES batches(batch_id) ON DELETE CASCADE,
    CHECK (photo_a < photo_b)
);

CREATE TABLE IF NOT EXISTS groups (
    batch_id  TEXT NOT NULL,
    group_idx INTEGER NOT NULL,
    photo_id  TEXT NOT NULL,
    PRIMARY KEY (batch_id, photo_id),
    FOREIGN KEY (batch_id) REFERENCES batches(batch_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS groups_order ON groups(batch_id, group_idx, photo_id);

CREATE TABLE IF NOT EXISTS predictions (
    batch_id         TEXT NOT NULL,
    photo_id         TEXT NOT NULL,
    known_species    TEXT NOT NULL,
    predicted_species TEXT NOT NULL,
    nearest_photo_id TEXT NOT NULL,
    distance         REAL NOT NULL,
    correct          INTEGER NOT NULL,
    PRIMARY KEY (batch_id, photo_id),
    FOREIGN KEY (batch_id) REFERENCES batches(batch_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS corrections (
    correction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id      TEXT NOT NULL,
    from_version  INTEGER NOT NULL,
    to_version    INTEGER NOT NULL,
    action        TEXT NOT NULL,
    photo_a       TEXT NOT NULL,
    photo_b       TEXT NOT NULL,
    recomputed_count INTEGER NOT NULL,
    created_at    TEXT NOT NULL,
    FOREIGN KEY (batch_id) REFERENCES batches(batch_id) ON DELETE CASCADE
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        str(db_path),
        timeout=30.0,
        isolation_level=None,  # Explicit BEGIN/COMMIT keeps transaction control local.
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def initialize(db_path: str | Path) -> None:
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    connection = connect(db_path)
    try:
        if str(db_path) != ":memory:":
            # WAL lets a reader see the last committed version while a
            # correction is still writing its new version.
            connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.executescript(SCHEMA)
    finally:
        connection.close()


@contextmanager
def write_transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Serialize writers immediately; readers are never blocked under WAL."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield connection
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    else:
        connection.execute("COMMIT")


@contextmanager
def read_snapshot(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Read all rows for one evaluation from a single consistent snapshot."""
    connection.execute("BEGIN")
    try:
        yield connection
    finally:
        connection.execute("ROLLBACK")
