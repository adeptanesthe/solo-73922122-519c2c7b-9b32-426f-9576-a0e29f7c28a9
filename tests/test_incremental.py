"""Property test: incremental correction state equals a full recomputation."""

from __future__ import annotations

import random

import numpy as np
import pytest

from app.database import connect, initialize
from app.grouping import build_groups, group_index
from app.models import BatchCreateRequest, CorrectionRequest, PhotoIn
from app.predictions import predict_nearest_outside
from app.service import ServiceError, apply_correction, create_batch, load_evaluation


def test_random_correction_sequence_matches_full_recompute(tmp_path):
    rng = random.Random(99)
    db_path = str(tmp_path / "fuzz.sqlite3")
    initialize(db_path)

    size = 14
    photo_ids = [f"q{i:02d}" for i in range(size)]
    photos = [
        PhotoIn(
            photo_id=photo_id,
            species=rng.choice(["a", "b", "c"]),
            features=[rng.randint(-4000, 4000) for _ in range(32)],
        )
        for photo_id in photo_ids
    ]
    pairs = [
        {"photo_a": "q00", "photo_b": "q01"},
        {"photo_a": "q02", "photo_b": "q03"},
        {"photo_a": "q03", "photo_b": "q04"},
    ]
    request = BatchCreateRequest.model_validate(
        {
            "batch_id": "fuzz",
            "photos": [photo.model_dump() for photo in photos],
            "pairs": pairs,
        }
    )
    create_batch(db_path, request)

    # Full-state oracle independent of the incremental path.
    features = np.asarray([photo.features for photo in photos], dtype=np.int64)
    species = {photo.photo_id: photo.species for photo in photos}
    edges = {(pair["photo_a"], pair["photo_b"]) for pair in pairs}
    version = 1

    for _ in range(40):
        a, b = rng.sample(photo_ids, 2)
        edge = (min(a, b), max(a, b))
        action = "remove" if edge in edges else "add"
        if not edges and action == "remove":
            action = "add"
            edge = (min(a, b), max(a, b))
        try:
            response = apply_correction(
                db_path,
                "fuzz",
                CorrectionRequest(
                    expected_version=version, action=action, photo_a=edge[0], photo_b=edge[1]
                ),
            )
        except ServiceError as exc:
            assert exc.status_code == 409
            continue
        version += 1
        if action == "add":
            edges.add(edge)
        else:
            edges.discard(edge)

        groups = build_groups(photo_ids, edges)
        mapping = group_index(groups)
        expected = (
            predict_nearest_outside(photo_ids, species, features, mapping, photo_ids)
            if len(groups) >= 2
            else {}
        )
        actual = load_evaluation(db_path, "fuzz")
        assert actual["version"] == version
        assert actual["group_count"] == len(groups)
        assert actual["evaluable"] is (len(groups) >= 2)
        assert {p["photo_id"]: p for p in actual["predictions"]} == expected
        assert response["recomputed_photo_ids"] == sorted(
            set(response["recomputed_photo_ids"])
        )

    # Stored edges are exactly the simulated set.
    connection = connect(db_path)
    try:
        stored = {
            (row["photo_a"], row["photo_b"])
            for row in connection.execute(
                "SELECT photo_a, photo_b FROM edges WHERE batch_id = 'fuzz'"
            )
        }
    finally:
        connection.close()
    assert stored == edges
