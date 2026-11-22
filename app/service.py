"""Batch ingestion, evaluation, and versioned corrections."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .database import connect, read_snapshot, write_transaction
from .grouping import build_groups, changed_membership_ids, group_index, normalize_pair
from .models import BatchCreateRequest, CorrectionRequest
from .predictions import predict_nearest_outside, summarize_groups

SINGLE_GROUP_MESSAGE = (
    "All photos belong to a single group, so no photo has an out-of-group "
    "neighbour. Accuracy cannot be evaluated."
)


class ServiceError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _new_batch_id() -> str:
    return uuid.uuid4().hex


def _load_photo_matrix(
    connection: sqlite3.Connection, batch_id: str
) -> tuple[list[str], dict[str, str], np.ndarray]:
    rows = connection.execute(
        "SELECT photo_id, species, features FROM photos WHERE batch_id = ? ORDER BY photo_id",
        (batch_id,),
    ).fetchall()
    photo_ids = [row["photo_id"] for row in rows]
    species = {row["photo_id"]: row["species"] for row in rows}
    features = np.asarray(
        [json.loads(row["features"]) for row in rows], dtype=np.int64
    ).reshape(len(rows), 32)
    return photo_ids, species, features


def _load_edges(connection: sqlite3.Connection, batch_id: str) -> set[tuple[str, str]]:
    rows = connection.execute(
        "SELECT photo_a, photo_b FROM edges WHERE batch_id = ?", (batch_id,)
    ).fetchall()
    return {(row["photo_a"], row["photo_b"]) for row in rows}


def _load_groups(connection: sqlite3.Connection, batch_id: str) -> list[list[str]]:
    rows = connection.execute(
        "SELECT group_idx, photo_id FROM groups WHERE batch_id = ? "
        "ORDER BY group_idx, photo_id",
        (batch_id,),
    ).fetchall()
    groups: list[list[str]] = []
    current_idx: int | None = None
    for row in rows:
        if row["group_idx"] != current_idx:
            groups.append([])
            current_idx = row["group_idx"]
        groups[-1].append(row["photo_id"])
    return groups


def _load_predictions(
    connection: sqlite3.Connection, batch_id: str
) -> dict[str, dict[str, Any]]:
    rows = connection.execute(
        "SELECT photo_id, known_species, predicted_species, nearest_photo_id, "
        "distance, correct FROM predictions WHERE batch_id = ? ORDER BY photo_id",
        (batch_id,),
    ).fetchall()
    return {
        row["photo_id"]: {
            "photo_id": row["photo_id"],
            "known_species": row["known_species"],
            "predicted_species": row["predicted_species"],
            "nearest_photo_id": row["nearest_photo_id"],
            "distance": row["distance"],
            "correct": bool(row["correct"]),
        }
        for row in rows
    }


def _compute_evaluation(
    photo_ids: Sequence[str],
    species: dict[str, str],
    features: np.ndarray,
    groups: Sequence[Sequence[str]],
) -> tuple[dict[str, dict[str, Any]], float | None, bool]:
    if len(groups) < 2:
        return {}, None, False

    mapping = group_index(groups)
    predictions = predict_nearest_outside(photo_ids, species, features, mapping, photo_ids)
    _, overall = summarize_groups(groups, predictions, True)
    return predictions, overall, True


def create_batch(db_path: str, request: BatchCreateRequest) -> dict[str, Any]:
    batch_id = (request.batch_id or _new_batch_id()).strip()
    if not batch_id:
        raise ServiceError(422, "invalid_batch_id", "batch_id must not be blank")

    # Cheap early rejection; the primary-key constraint below still guards races.
    existing = connect(db_path)
    try:
        if existing.execute(
            "SELECT 1 FROM batches WHERE batch_id = ?", (batch_id,)
        ).fetchone():
            raise ServiceError(409, "batch_exists", f"batch {batch_id!r} already exists")
    finally:
        existing.close()

    photo_ids = sorted(photo.photo_id for photo in request.photos)
    species = {photo.photo_id: photo.species for photo in request.photos}
    features = np.asarray(
        [photo.features for photo in sorted(request.photos, key=lambda p: p.photo_id)],
        dtype=np.int64,
    )
    edges = {normalize_pair(pair.photo_a, pair.photo_b) for pair in request.pairs}
    groups = build_groups(photo_ids, edges)
    predictions, overall, evaluable = _compute_evaluation(
        photo_ids, species, features, groups
    )
    created_at = _now()
    reason = None if evaluable else "single_group"

    connection = connect(db_path)
    try:
        with write_transaction(connection) as tx:
            try:
                tx.execute(
                    "INSERT INTO batches (batch_id, created_at, current_version, "
                    "version_created_at, photo_count, group_count, evaluable, "
                    "overall_accuracy, reason) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?)",
                    (
                        batch_id,
                        created_at,
                        created_at,
                        len(photo_ids),
                        len(groups),
                        1 if evaluable else 0,
                        overall,
                        reason,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                if "batches.batch_id" in str(exc):
                    raise ServiceError(
                        409, "batch_exists", f"batch {batch_id!r} already exists"
                    ) from exc
                raise

            tx.executemany(
                "INSERT INTO photos (batch_id, photo_id, species, features) "
                "VALUES (?, ?, ?, ?)",
                [
                    (
                        batch_id,
                        photo.photo_id,
                        photo.species,
                        json.dumps(photo.features, separators=(",", ":")),
                    )
                    for photo in sorted(request.photos, key=lambda p: p.photo_id)
                ],
            )
            tx.executemany(
                "INSERT INTO edges (batch_id, photo_a, photo_b) VALUES (?, ?, ?)",
                [(batch_id, photo_a, photo_b) for photo_a, photo_b in sorted(edges)],
            )
            _write_groups(tx, batch_id, groups)
            _replace_predictions(tx, batch_id, predictions)
    finally:
        connection.close()

    return load_evaluation(db_path, batch_id)


def _write_groups(
    connection: sqlite3.Connection, batch_id: str, groups: Sequence[Sequence[str]]
) -> None:
    connection.execute("DELETE FROM groups WHERE batch_id = ?", (batch_id,))
    connection.executemany(
        "INSERT INTO groups (batch_id, group_idx, photo_id) VALUES (?, ?, ?)",
        [
            (batch_id, index, photo_id)
            for index, members in enumerate(groups)
            for photo_id in members
        ],
    )


def _replace_predictions(
    connection: sqlite3.Connection,
    batch_id: str,
    predictions: dict[str, dict[str, Any]],
) -> None:
    connection.execute("DELETE FROM predictions WHERE batch_id = ?", (batch_id,))
    connection.executemany(
        "INSERT INTO predictions (batch_id, photo_id, known_species, "
        "predicted_species, nearest_photo_id, distance, correct) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                batch_id,
                prediction["photo_id"],
                prediction["known_species"],
                prediction["predicted_species"],
                prediction["nearest_photo_id"],
                prediction["distance"],
                1 if prediction["correct"] else 0,
            )
            for prediction in predictions.values()
        ],
    )


def _upsert_predictions(
    connection: sqlite3.Connection,
    batch_id: str,
    predictions: dict[str, dict[str, Any]],
) -> None:
    connection.executemany(
        "INSERT INTO predictions (batch_id, photo_id, known_species, "
        "predicted_species, nearest_photo_id, distance, correct) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(batch_id, photo_id) DO UPDATE SET "
        "known_species = excluded.known_species, "
        "predicted_species = excluded.predicted_species, "
        "nearest_photo_id = excluded.nearest_photo_id, "
        "distance = excluded.distance, correct = excluded.correct",
        [
            (
                batch_id,
                prediction["photo_id"],
                prediction["known_species"],
                prediction["predicted_species"],
                prediction["nearest_photo_id"],
                prediction["distance"],
                1 if prediction["correct"] else 0,
            )
            for prediction in predictions.values()
        ],
    )


def load_evaluation(db_path: str, batch_id: str) -> dict[str, Any]:
    connection = connect(db_path)
    try:
        with read_snapshot(connection) as snapshot:
            batch = snapshot.execute(
                "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if batch is None:
                raise ServiceError(404, "batch_not_found", f"batch {batch_id!r} not found")
            groups = _load_groups(snapshot, batch_id)
            predictions_by_id = _load_predictions(snapshot, batch_id)

        evaluable = bool(batch["evaluable"])
        reasons = {"single_group": SINGLE_GROUP_MESSAGE}
        reason_key = batch["reason"]
        group_output, overall = summarize_groups(
            groups, predictions_by_id, evaluable
        )
        return {
            "batch_id": batch_id,
            "version": batch["current_version"],
            "photo_count": batch["photo_count"],
            "group_count": batch["group_count"],
            "evaluable": evaluable,
            "reason": reason_key,
            "message": reasons.get(reason_key) if reason_key else None,
            "overall_accuracy": batch["overall_accuracy"],
            "groups": group_output,
            "predictions": [
                predictions_by_id[photo_id]
                for photo_id in sorted(predictions_by_id)
            ],
            "created_at": batch["created_at"],
            "version_created_at": batch["version_created_at"],
        }
    finally:
        connection.close()


def apply_correction(
    db_path: str, batch_id: str, request: CorrectionRequest
) -> dict[str, Any]:
    connection = connect(db_path)
    try:
        with write_transaction(connection) as tx:
            batch = tx.execute(
                "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if batch is None:
                raise ServiceError(404, "batch_not_found", f"batch {batch_id!r} not found")
            if batch["current_version"] != request.expected_version:
                raise ServiceError(
                    409,
                    "version_conflict",
                    "expected_version does not match the current batch version",
                )

            photo_ids, species, features = _load_photo_matrix(tx, batch_id)
            known_ids = set(photo_ids)
            if request.photo_a not in known_ids or request.photo_b not in known_ids:
                raise ServiceError(
                    422,
                    "unknown_photo",
                    "both correction endpoints must be photos in this batch",
                )

            edge = normalize_pair(request.photo_a, request.photo_b)
            edges = _load_edges(tx, batch_id)
            if request.action == "add":
                if edge in edges:
                    raise ServiceError(
                        409, "edge_exists", "that photo pair is already linked"
                    )
                edges.add(edge)
            else:
                if edge not in edges:
                    raise ServiceError(
                        409, "edge_not_found", "that photo pair is not linked"
                    )
                edges.remove(edge)

            old_groups = _load_groups(tx, batch_id)
            new_groups = build_groups(photo_ids, edges)
            changed_ids = changed_membership_ids(old_groups, new_groups, photo_ids)

            evaluable = len(new_groups) >= 2
            if evaluable:
                mapping = group_index(new_groups)
                recomputed = predict_nearest_outside(
                    photo_ids, species, features, mapping, changed_ids
                )
                old_predictions = _load_predictions(tx, batch_id)
                combined = {
                    photo_id: prediction
                    for photo_id, prediction in old_predictions.items()
                    if photo_id not in recomputed
                }
                combined.update(recomputed)
                _, overall = summarize_groups(new_groups, combined, True)
            else:
                recomputed = {}
                combined = {}
                overall = None

            if request.action == "add":
                tx.execute(
                    "INSERT INTO edges (batch_id, photo_a, photo_b) VALUES (?, ?, ?)",
                    (batch_id, edge[0], edge[1]),
                )
            else:
                tx.execute(
                    "DELETE FROM edges WHERE batch_id = ? AND photo_a = ? AND photo_b = ?",
                    (batch_id, edge[0], edge[1]),
                )

            _write_groups(tx, batch_id, new_groups)
            if evaluable:
                _upsert_predictions(tx, batch_id, recomputed)
            else:
                tx.execute("DELETE FROM predictions WHERE batch_id = ?", (batch_id,))

            new_version = request.expected_version + 1
            version_created_at = _now()
            reason = None if evaluable else "single_group"
            updated = tx.execute(
                "UPDATE batches SET current_version = ?, version_created_at = ?, "
                "group_count = ?, evaluable = ?, overall_accuracy = ?, reason = ? "
                "WHERE batch_id = ? AND current_version = ?",
                (
                    new_version,
                    version_created_at,
                    len(new_groups),
                    1 if evaluable else 0,
                    overall,
                    reason,
                    batch_id,
                    request.expected_version,
                ),
            )
            if updated.rowcount != 1:
                raise ServiceError(
                    409,
                    "version_conflict",
                    "another correction changed this version first",
                )
            tx.execute(
                "INSERT INTO corrections (batch_id, from_version, to_version, action, "
                "photo_a, photo_b, recomputed_count, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    batch_id,
                    request.expected_version,
                    new_version,
                    request.action,
                    edge[0],
                    edge[1],
                    len(changed_ids),
                    version_created_at,
                ),
            )
    finally:
        connection.close()

    return {
        "batch_id": batch_id,
        "previous_version": request.expected_version,
        "version": request.expected_version + 1,
        "action": request.action,
        "photo_a": edge[0],
        "photo_b": edge[1],
        "recomputed_photo_ids": changed_ids,
        "photo_count": len(photo_ids),
        "group_count": len(new_groups),
        "evaluable": evaluable,
        "reason": reason,
        "message": SINGLE_GROUP_MESSAGE if not evaluable else None,
        "overall_accuracy": overall,
        "version_created_at": version_created_at,
    }
