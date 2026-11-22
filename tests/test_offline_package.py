"""Tests for the offline reference-package API."""

from __future__ import annotations

import random
import threading
import time

import numpy as np
import pytest

from app.offline_package import (
    NO_DISABLED,
    solve_minimum_package,
    squared_distance_matrix,
)


def vec(x: int = 0, y: int = 0) -> list[int]:
    features = [0] * 32
    features[0] = x
    features[1] = y
    return features


def demo_batch(batch_id: str = "offline-demo") -> dict:
    # r1/r2 are one camera group (deer), r3/r4 another (fox), r5 a singleton
    # otter group. v1 sits next to r1; v2 sits next to r5.
    photos = [
        {"photo_id": "r1", "species": "deer", "features": vec(0, 0)},
        {"photo_id": "r2", "species": "deer", "features": vec(100, 0)},
        {"photo_id": "r3", "species": "fox", "features": vec(10, 0)},
        {"photo_id": "r4", "species": "fox", "features": vec(110, 0)},
        {"photo_id": "r5", "species": "otter", "features": vec(0, 100)},
        {"photo_id": "v1", "species": "fox", "features": vec(1, 0)},
        {"photo_id": "v2", "species": "otter", "features": vec(0, 95)},
    ]
    pairs = [
        {"photo_a": "r1", "photo_b": "r2"},
        {"photo_a": "r3", "photo_b": "r4"},
    ]
    return {"batch_id": batch_id, "photos": photos, "pairs": pairs}


PACKAGE_REQUEST = {
    "expected_version": 1,
    "candidate_photo_ids": ["r1", "r2", "r3", "r4", "r5"],
    "verification_photo_ids": ["v1", "v2"],
    "capacity": 3,
}


def test_package_selects_minimum_photos_needed_for_every_outage(client):
    assert client.post("/batches", json=demo_batch()).status_code == 201
    response = client.post("/batches/offline-demo/offline-package", json=PACKAGE_REQUEST)
    assert response.status_code == 200, response.text
    body = response.json()

    # r1 covers the deer verdict (and the fallback when group C is off),
    # r3 covers the fox verdict once the deer camera group is disabled,
    # r5 covers the otter verdict.
    assert [p["photo_id"] for p in body["package_photos"]] == ["r1", "r3", "r5"]
    assert body["package_size"] == 3
    assert body["version"] == 1
    assert body["expected_version"] == 1

    by_scenario = {(s["verification_photo_id"], s["scenario"], s["disabled_group_id"]): s
                   for s in body["scenarios"]}
    v1_all = by_scenario[("v1", "all_enabled", None)]
    assert v1_all["full_nearest_photo_id"] == "r1"
    assert v1_all["package_nearest_photo_id"] == "r1"
    v1_no_deer = by_scenario[("v1", "group_disabled", "r1")]
    assert v1_no_deer["full_nearest_photo_id"] == "r3"
    assert v1_no_deer["full_species"] == "fox"
    assert v1_no_deer["package_nearest_photo_id"] == "r3"
    v2_no_otter = by_scenario[("v2", "group_disabled", "r5")]
    assert v2_no_otter["full_nearest_photo_id"] == "r1"
    assert v2_no_otter["package_species"] == v2_no_otter["full_species"] == "deer"

    # Two verification photos times (all-enabled + three group outages).
    assert len(body["scenarios"]) == 2 * 4


def test_package_photos_carry_features_species_and_group_label(client):
    assert client.post("/batches", json=demo_batch()).status_code == 201
    body = client.post(
        "/batches/offline-demo/offline-package", json=PACKAGE_REQUEST
    ).json()
    by_id = {p["photo_id"]: p for p in body["package_photos"]}
    assert by_id["r1"]["features"] == vec(0, 0)
    assert by_id["r1"]["species"] == "deer"
    assert by_id["r1"]["group_id"] == "r1"  # smallest candidate ID in the group
    assert by_id["r3"]["group_id"] == "r3"
    assert by_id["r5"]["group_id"] == "r5"

    groups = {p["photo_id"]: p["group_id"] for p in body["candidate_groups"]}
    assert groups == {"r1": "r1", "r2": "r1", "r3": "r3", "r4": "r3", "r5": "r5"}


def test_capacity_shortfall_returns_required_size(client):
    assert client.post("/batches", json=demo_batch()).status_code == 201
    response = client.post(
        "/batches/offline-demo/offline-package",
        json={**PACKAGE_REQUEST, "capacity": 2},
    )
    assert response.status_code == 409
    error = response.json()
    assert error["error"] == "capacity_exceeded"
    assert error["details"] == {"capacity": 2, "required_size": 3, "version": 1}


def test_stale_expected_version_is_rejected(client):
    assert client.post("/batches", json=demo_batch()).status_code == 201
    response = client.post(
        "/batches/offline-demo/offline-package",
        json={**PACKAGE_REQUEST, "expected_version": 7},
    )
    assert response.status_code == 409
    assert response.json()["error"] == "version_conflict"
    assert response.json()["details"]["current_version"] == 1


def test_unknown_batch_and_unknown_photos(client):
    assert client.post("/batches", json=demo_batch()).status_code == 201
    assert client.post(
        "/batches/missing/offline-package", json=PACKAGE_REQUEST
    ).status_code == 404

    bad = client.post(
        "/batches/offline-demo/offline-package",
        json={**PACKAGE_REQUEST, "candidate_photo_ids": ["r1", "ghost"]},
    )
    assert bad.status_code == 422
    assert bad.json()["details"]["unknown_photo_ids"] == ["ghost"]


def test_request_validation_limits_and_overlap(client):
    assert client.post("/batches", json=demo_batch()).status_code == 201
    base = PACKAGE_REQUEST
    too_many_candidates = {
        **base,
        "candidate_photo_ids": [f"r{i}" for i in range(17)],
    }
    assert client.post(
        "/batches/offline-demo/offline-package", json=too_many_candidates
    ).status_code == 422

    too_many_verifications = {
        **base,
        "verification_photo_ids": [f"v{i}" for i in range(41)],
    }
    assert client.post(
        "/batches/offline-demo/offline-package", json=too_many_verifications
    ).status_code == 422

    overlap = {
        **base,
        "candidate_photo_ids": ["r1", "r2"],
        "verification_photo_ids": ["r1"],
    }
    assert client.post(
        "/batches/offline-demo/offline-package", json=overlap
    ).status_code == 422

    zero_capacity = {**base, "capacity": 0}
    assert client.post(
        "/batches/offline-demo/offline-package", json=zero_capacity
    ).status_code == 422


def test_no_reference_on_both_sides_when_every_reference_is_unreachable(client):
    # One candidate group holding every candidate and a verification photo in
    # a different group. With the group enabled the photo has a reference;
    # once the only candidate group is disabled neither library has one, so
    # both must answer "no reference" and the outage adds no required photo.
    photos = [
        {"photo_id": "r1", "species": "deer", "features": vec(0, 0)},
        {"photo_id": "r2", "species": "deer", "features": vec(1, 0)},
        {"photo_id": "v1", "species": "fox", "features": vec(50, 50)},
    ]
    payload = {
        "batch_id": "no-ref",
        "photos": photos,
        "pairs": [{"photo_a": "r1", "photo_b": "r2"}],
    }
    assert client.post("/batches", json=payload).status_code == 201
    body = client.post(
        "/batches/no-ref/offline-package",
        json={
            "expected_version": 1,
            "candidate_photo_ids": ["r1", "r2"],
            "verification_photo_ids": ["v1"],
            "capacity": 2,
        },
    ).json()
    # The all-enabled scenario still needs one deer; the outage scenario
    # contributes nothing, so a single smallest-ID reference is enough.
    assert body["package_size"] == 1
    assert [p["photo_id"] for p in body["package_photos"]] == ["r1"]
    disabled = next(s for s in body["scenarios"] if s["scenario"] == "group_disabled")
    assert disabled["full_nearest_photo_id"] is None
    assert disabled["full_species"] is None
    assert disabled["package_nearest_photo_id"] is None
    assert disabled["package_species"] is None
    enabled = next(s for s in body["scenarios"] if s["scenario"] == "all_enabled")
    assert enabled["full_species"] == enabled["package_species"] == "deer"


def test_lexicographic_tie_break_among_equal_size_packages():
    # Every candidate is its own group and every candidate is the same
    # species. The all-enabled nearest is a, but once a's own group is the
    # simulated outage another deer is required. Size two is minimal; among
    # the valid pairs {a,b}, {a,c}, {a,d} the smallest sorted ID tuple wins.
    candidate_ids = ["a", "b", "c", "d"]
    verification_ids = ["v1"]
    candidate_group = [0, 1, 2, 3]
    verification_group = [4]
    candidate_species = ["deer", "deer", "deer", "deer"]
    features = np.asarray(
        [
            [0, 0],    # a
            [5, 0],    # b
            [0, 100],  # c
            [5, 100],  # d
            [1, 0],    # v1: nearest a
        ],
        dtype=np.int64,
    )
    distances = squared_distance_matrix(features)[1:, :4]
    selected, verdicts = solve_minimum_package(
        candidate_ids,
        verification_ids,
        candidate_group,
        verification_group,
        candidate_species,
        distances,
        [0, 1, 2, 3],
    )
    assert [candidate_ids[i] for i in selected] == ["a", "b"]
    for verdict in verdicts:
        full = verdict["full_index"]
        package = verdict["package_index"]
        full_species = None if full is None else candidate_species[full]
        package_species = None if package is None else candidate_species[package]
        assert full_species == package_species


def test_lexicographic_tie_break_among_equal_size_multi_species_packages():
    # Two required species mean the minimum size is two; among the valid
    # size-two tuples the lexicographically smallest sorted ID tuple wins.
    candidate_ids = ["a", "b", "c", "d"]
    verification_ids = ["v1"]
    candidate_group = [0, 1, 2, 3]
    verification_group = [4]
    candidate_species = ["deer", "fox", "deer", "fox"]
    features = np.asarray(
        [
            [0, 0],    # a deer
            [5, 0],    # b fox
            [0, 100],  # c deer
            [5, 100],  # d fox
            [1, 0],    # v1
        ],
        dtype=np.int64,
    )
    distances = squared_distance_matrix(features)[1:, :4]
    selected, _ = solve_minimum_package(
        candidate_ids,
        verification_ids,
        candidate_group,
        verification_group,
        candidate_species,
        distances,
        [0, 1, 2, 3],
    )
    assert [candidate_ids[i] for i in selected] == ["a", "b"]


def test_correction_changing_selected_relationships_is_reported_as_conflict(
    client, monkeypatch
):
    import app.service as service

    assert client.post("/batches", json=demo_batch("race1")).status_code == 201
    original = service.solve_minimum_package

    def slow(*args, **kwargs):
        time.sleep(0.6)
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "solve_minimum_package", slow)
    outcome: dict = {}

    def request_package() -> None:
        outcome["response"] = client.post(
            "/batches/race1/offline-package", json=PACKAGE_REQUEST
        )

    thread = threading.Thread(target=request_package)
    thread.start()
    time.sleep(0.2)
    correction = client.post(
        "/batches/race1/corrections",
        json={"expected_version": 1, "action": "add", "photo_a": "r1", "photo_b": "r3"},
    )
    thread.join()
    assert correction.status_code == 200
    response = outcome["response"]
    assert response.status_code == 409
    assert response.json()["error"] == "group_relationship_changed"
    assert response.json()["details"]["current_version"] == 2


def test_unrelated_correction_keeps_package_and_stamps_new_version(
    client, monkeypatch
):
    import app.service as service

    assert client.post("/batches", json=demo_batch("race2")).status_code == 201
    original = service.solve_minimum_package

    def slow(*args, **kwargs):
        time.sleep(0.6)
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "solve_minimum_package", slow)
    outcome: dict = {}

    def request_package() -> None:
        outcome["response"] = client.post(
            "/batches/race2/offline-package", json=PACKAGE_REQUEST
        )

    thread = threading.Thread(target=request_package)
    thread.start()
    time.sleep(0.2)
    # Only touches the two verification photos; selected candidates keep the
    # exact same same-group relationships.
    correction = client.post(
        "/batches/race2/corrections",
        json={"expected_version": 1, "action": "add", "photo_a": "v1", "photo_b": "v2"},
    )
    thread.join()
    assert correction.status_code == 200
    response = outcome["response"]
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["expected_version"] == 1
    assert body["version"] == 2
    assert [p["photo_id"] for p in body["package_photos"]] == ["r1", "r3", "r5"]


def test_corrections_remain_available_while_package_is_solving(
    client, monkeypatch
):
    import app.service as service

    assert client.post("/batches", json=demo_batch("race3")).status_code == 201
    original = service.solve_minimum_package

    def slow(*args, **kwargs):
        time.sleep(0.6)
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "solve_minimum_package", slow)
    outcome: dict = {}

    def request_package() -> None:
        outcome["response"] = client.post(
            "/batches/race3/offline-package", json=PACKAGE_REQUEST
        )

    thread = threading.Thread(target=request_package)
    thread.start()
    time.sleep(0.2)
    # The package request holds no write lock, so the correction commits now.
    correction = client.post(
        "/batches/race3/corrections",
        json={"expected_version": 1, "action": "add", "photo_a": "v1", "photo_b": "v2"},
    )
    thread.join()
    assert correction.status_code == 200
    assert outcome["response"].status_code == 200
    assert client.get("/batches/race3/evaluation").json()["version"] == 2


def test_solver_matches_brute_force_on_random_instances():
    rng = random.Random(2026)
    for trial in range(60):
        candidate_count = rng.randint(1, 9)
        verification_count = rng.randint(1, 6)
        candidate_ids = sorted(f"c{i:02d}" for i in range(candidate_count))
        verification_ids = sorted(f"v{i:02d}" for i in range(verification_count))
        group_count = rng.randint(1, candidate_count)
        candidate_group = [rng.randrange(group_count) for _ in candidate_ids]
        verification_group = [
            rng.randrange(group_count + 2) for _ in verification_ids
        ]
        candidate_species = [
            rng.choice(["deer", "fox", "otter"]) for _ in candidate_ids
        ]
        features = np.asarray(
            [[rng.randint(-20, 20), rng.randint(-20, 20)]
             for _ in candidate_ids + verification_ids],
            dtype=np.int64,
        )
        distances = squared_distance_matrix(features)[candidate_count:, :candidate_count]
        distinct_groups = sorted(set(candidate_group))
        selected, verdicts = solve_minimum_package(
            candidate_ids,
            verification_ids,
            candidate_group,
            verification_group,
            candidate_species,
            distances,
            distinct_groups,
        )

        def verdict_for(subset: set[int], v_index: int, disabled: int):
            own_group = verification_group[v_index]
            usable = [
                i for i in range(candidate_count)
                if candidate_group[i] != own_group
                and candidate_group[i] != disabled
                and i in subset
            ]
            if not usable:
                return None
            best = min(usable, key=lambda i: (int(distances[v_index, i]), candidate_ids[i]))
            return candidate_species[best]

        def full_verdict(v_index: int, disabled: int):
            own_group = verification_group[v_index]
            usable = [
                i for i in range(candidate_count)
                if candidate_group[i] != own_group
                and candidate_group[i] != disabled
            ]
            if not usable:
                return None
            best = min(usable, key=lambda i: (int(distances[v_index, i]), candidate_ids[i]))
            return candidate_species[best]

        scenarios = [NO_DISABLED, *distinct_groups]
        # Every returned verdict must match the full library.
        for verdict in verdicts:
            v_index = verification_ids.index(verdict["verification_photo_id"])
            full = full_verdict(v_index, verdict["disabled_group"])
            package = verdict_for(set(selected), v_index, verdict["disabled_group"])
            assert full == package, (trial, verdict)

        # Brute-force minimum: no smaller subset, and no lexicographically
        # earlier same-size subset, may satisfy every scenario.
        from itertools import combinations

        def satisfies(subset: set[int]) -> bool:
            for v_index in range(verification_count):
                for disabled in scenarios:
                    if full_verdict(v_index, disabled) != verdict_for(
                        subset, v_index, disabled
                    ):
                        return False
            return True

        assert satisfies(set(selected))
        winner = tuple(candidate_ids[i] for i in selected)
        for size in range(len(winner) + 1):
            for combo in combinations(range(candidate_count), size):
                if size < len(winner) or tuple(candidate_ids[i] for i in combo) < winner:
                    assert not satisfies(set(combo))


def test_empty_package_when_every_scenario_is_mutually_unreachable(client):
    # All photos in one group: every verification photo's own group contains
    # every candidate, so no scenario ever has a usable reference. Both
    # libraries must answer "no reference" everywhere and the empty package
    # is the minimum.
    photos = [
        {"photo_id": "r1", "species": "deer", "features": vec(0, 0)},
        {"photo_id": "r2", "species": "fox", "features": vec(5, 0)},
        {"photo_id": "v1", "species": "otter", "features": vec(9, 0)},
    ]
    payload = {
        "batch_id": "empty-pkg",
        "photos": photos,
        "pairs": [
            {"photo_a": "r1", "photo_b": "r2"},
            {"photo_a": "r2", "photo_b": "v1"},
        ],
    }
    assert client.post("/batches", json=payload).status_code == 201
    body = client.post(
        "/batches/empty-pkg/offline-package",
        json={
            "expected_version": 1,
            "candidate_photo_ids": ["r1", "r2"],
            "verification_photo_ids": ["v1"],
            "capacity": 2,
        },
    ).json()
    assert body["package_size"] == 0
    assert body["package_photos"] == []
    assert all(s["full_species"] is None and s["package_species"] is None
               for s in body["scenarios"])


def test_group_labels_use_snapshot_groups_even_after_unrelated_correction(
    client, monkeypatch
):
    import app.service as service

    assert client.post("/batches", json=demo_batch("labels")).status_code == 201
    original = service.solve_minimum_package

    def slow(*args, **kwargs):
        time.sleep(0.6)
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "solve_minimum_package", slow)
    outcome: dict = {}

    def request_package() -> None:
        outcome["response"] = client.post(
            "/batches/labels/offline-package", json=PACKAGE_REQUEST
        )

    thread = threading.Thread(target=request_package)
    thread.start()
    time.sleep(0.2)
    # Merges the two verification photos only; every candidate group keeps
    # its snapshot composition and therefore its snapshot label.
    correction = client.post(
        "/batches/labels/corrections",
        json={"expected_version": 1, "action": "add", "photo_a": "v1", "photo_b": "v2"},
    )
    thread.join()
    assert correction.status_code == 200
    body = outcome["response"].json()
    assert body["version"] == 2
    labels = {p["photo_id"]: p["group_id"] for p in body["candidate_groups"]}
    assert labels == {"r1": "r1", "r2": "r1", "r3": "r3", "r4": "r3", "r5": "r5"}


def test_sixteen_candidates_are_supported_and_fast():
    rng = random.Random(11)
    candidate_count = 16
    verification_count = 40
    candidate_ids = sorted(f"c{i:02d}" for i in range(candidate_count))
    verification_ids = sorted(f"v{i:02d}" for i in range(verification_count))
    candidate_group = [rng.randrange(6) for _ in candidate_ids]
    verification_group = [rng.randrange(8) for _ in verification_ids]
    candidate_species = [rng.choice(["a", "b", "c"]) for _ in candidate_ids]
    features = np.asarray(
        [rng.sample(range(-300, 300), 2) for _ in candidate_ids + verification_ids],
        dtype=np.int64,
    )
    distances = squared_distance_matrix(features)[candidate_count:, :candidate_count]
    started = time.perf_counter()
    selected, verdicts = solve_minimum_package(
        candidate_ids,
        verification_ids,
        candidate_group,
        verification_group,
        candidate_species,
        distances,
        sorted(set(candidate_group)),
    )
    elapsed = time.perf_counter() - started
    assert elapsed < 5.0
    assert len(verdicts) == verification_count * (1 + len(set(candidate_group)))
    assert all(0 <= i < candidate_count for i in selected)
