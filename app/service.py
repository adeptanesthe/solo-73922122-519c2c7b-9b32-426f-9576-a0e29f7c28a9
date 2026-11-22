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
from .models import BatchCreateRequest, CorrectionRequest, OfflinePackageRequest
from .offline_package import (
    NO_DISABLED,
    solve_minimum_package,
    squared_distance_matrix,
)
from .predictions import predict_nearest_outside, summarize_groups

SINGLE_GROUP_MESSAGE = (
    "All photos belong to a single group, so no photo has an out-of-group "
    "neighbour. Accuracy cannot be evaluated."
)


class ServiceError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


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


def _group_label_for(
    group_of: dict[str, int], groups: Sequence[Sequence[str]], photo_id: str
) -> str | None:
    """Camera-group label = smallest candidate reference ID in the group."""
    index = group_of.get(photo_id)
    if index is None:
        return None
    return groups[index][0]


def build_offline_package(
    db_path: str, batch_id: str, request: OfflinePackageRequest
) -> dict[str, Any]:
    """Select the minimum offline reference package for one request.

    The whole computation reads a single WAL snapshot and never takes a write
    lock, so reviewer corrections can still commit while the package is being
    solved. Same-group relationships between the selected photos are then
    re-checked against the latest committed version: any change is reported as
    a conflict, otherwise the answer is stamped with that newer version.
    """
    connection = connect(db_path)
    try:
        with read_snapshot(connection) as snapshot:
            batch = snapshot.execute(
                "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if batch is None:
                raise ServiceError(
                    404, "batch_not_found", f"batch {batch_id!r} not found"
                )
            snapshot_version = batch["current_version"]
            if snapshot_version != request.expected_version:
                raise ServiceError(
                    409,
                    "version_conflict",
                    "expected_version does not match the current batch version",
                    details={
                        "expected_version": request.expected_version,
                        "current_version": snapshot_version,
                    },
                )

            rows = snapshot.execute(
                "SELECT photo_id, species, features FROM photos "
                "WHERE batch_id = ? ORDER BY photo_id",
                (batch_id,),
            ).fetchall()
            all_photo_ids = [row["photo_id"] for row in rows]
            species_by_id = {row["photo_id"]: row["species"] for row in rows}
            features_by_id = {
                row["photo_id"]: json.loads(row["features"]) for row in rows
            }
            groups = _load_groups(snapshot, batch_id)

        known = set(all_photo_ids)
        unknown = sorted(
            photo_id
            for photo_id in (*request.candidate_photo_ids, *request.verification_photo_ids)
            if photo_id not in known
        )
        if unknown:
            raise ServiceError(
                422,
                "unknown_photo",
                "every candidate and verification photo must exist in this batch",
                details={"unknown_photo_ids": unknown},
            )

        candidate_ids = sorted(request.candidate_photo_ids)
        verification_ids = sorted(request.verification_photo_ids)
        group_of = group_index(groups)

        candidate_group_indices = sorted({group_of[photo_id] for photo_id in candidate_ids})
        candidate_group = [group_of[photo_id] for photo_id in candidate_ids]
        verification_group = [group_of[photo_id] for photo_id in verification_ids]
        candidate_species = [species_by_id[photo_id] for photo_id in candidate_ids]

        involved_ids = candidate_ids + verification_ids
        involved_matrix = np.asarray(
            [features_by_id[photo_id] for photo_id in involved_ids], dtype=np.int64
        )
        full_distances = squared_distance_matrix(involved_matrix)
        n_candidates = len(candidate_ids)
        # Rows 0..n-1 are candidates; the remaining rows are the verification
        # photos in the same sorted order as verification_ids.
        distances = full_distances[n_candidates:, :n_candidates]

        selected, verdicts = solve_minimum_package(
            candidate_ids,
            verification_ids,
            candidate_group,
            verification_group,
            candidate_species,
            distances,
            candidate_group_indices,
        )
        selected_ids = [candidate_ids[index] for index in selected]

        # Re-check same-group relationships among the selected photos against
        # the latest committed state. The snapshot transaction has ended, so a
        # correction committing in between is visible here.
        latest = connection.execute(
            "SELECT current_version, version_created_at FROM batches WHERE batch_id = ?",
            (batch_id,),
        ).fetchone()
        if latest is None:
            raise ServiceError(404, "batch_not_found", f"batch {batch_id!r} not found")
        recheck_version = latest["current_version"]
        recheck_groups = _load_groups(connection, batch_id)
        recheck_group_of = group_index(recheck_groups)

        def signature(photo_ids: Sequence[str], mapping: dict[str, int]) -> set[frozenset[str]]:
            return {
                frozenset((a, b))
                for a in photo_ids
                for b in photo_ids
                if a < b and mapping[a] == mapping[b]
            }

        if signature(selected_ids, group_of) != signature(selected_ids, recheck_group_of):
            raise ServiceError(
                409,
                "group_relationship_changed",
                "a correction changed same-group relationships between selected "
                "photos while the package was being solved; retry against the "
                "returned version",
                details={
                    "expected_version": request.expected_version,
                    "current_version": recheck_version,
                },
            )

        required_size = len(selected_ids)
        if required_size > request.capacity:
            raise ServiceError(
                409,
                "capacity_exceeded",
                "the smallest package that survives every group outage exceeds "
                "the requested capacity",
                details={
                    "capacity": request.capacity,
                    "required_size": required_size,
                    "version": recheck_version,
                },
            )

        def package_photo(
            photo_id: str,
            mapping: dict[str, int],
            labeled_groups: Sequence[Sequence[str]],
        ) -> dict[str, Any]:
            return {
                "photo_id": photo_id,
                "features": features_by_id[photo_id],
                "species": species_by_id[photo_id],
                "group_id": _group_label_for(mapping, labeled_groups, photo_id),
            }

        scenario_output: list[dict[str, Any]] = []
        for verdict in verdicts:
            full_index = verdict["full_index"]
            package_index = verdict["package_index"]
            disabled = verdict["disabled_group"]
            scenario_output.append(
                {
                    "scenario": "all_enabled" if disabled == NO_DISABLED else "group_disabled",
                    "disabled_group_id": (
                        None
                        if disabled == NO_DISABLED
                        else groups[disabled][0]
                    ),
                    "verification_photo_id": verdict["verification_photo_id"],
                    "full_nearest_photo_id": (
                        None if full_index is None else candidate_ids[full_index]
                    ),
                    "full_species": (
                        None
                        if full_index is None
                        else candidate_species[full_index]
                    ),
                    "package_nearest_photo_id": (
                        None if package_index is None else candidate_ids[package_index]
                    ),
                    "package_species": (
                        None
                        if package_index is None
                        else candidate_species[package_index]
                    ),
                }
            )

        return {
            "batch_id": batch_id,
            "expected_version": request.expected_version,
            "version": recheck_version,
            "capacity": request.capacity,
            "package_size": required_size,
            "candidate_photo_ids": candidate_ids,
            "candidate_groups": [
                package_photo(photo_id, group_of, groups)
                for photo_id in candidate_ids
            ],
            "verification_photo_ids": verification_ids,
            "package_photos": [
                package_photo(photo_id, recheck_group_of, recheck_groups)
                for photo_id in selected_ids
            ],
            "scenarios": scenario_output,
            "version_created_at": latest["version_created_at"],
        }
    finally:
        connection.close()
