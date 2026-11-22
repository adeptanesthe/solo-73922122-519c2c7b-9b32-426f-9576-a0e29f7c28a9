from __future__ import annotations

import copy
import json
import threading
from pathlib import Path

import numpy as np
import pytest

from app.predictions import predict_nearest_outside
from app.grouping import build_groups, changed_membership_ids

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = json.loads((ROOT / "examples" / "create_batch.json").read_text())


def demo_payload(batch_id="b1"):
    payload = copy.deepcopy(EXAMPLE)
    payload["batch_id"] = batch_id
    return payload


def test_create_batch_returns_predictions_and_equal_weight_groups(client):
    response = client.post("/batches", json=demo_payload())
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["version"] == 1
    assert body["photo_count"] == 10
    assert body["group_count"] == 6
    assert body["evaluable"] is True
    by_id = {p["photo_id"]: p for p in body["predictions"]}
    # Same-session partner must never be the nearest neighbour.
    assert by_id["p01"]["nearest_photo_id"] not in {"p02"}
    assert by_id["p02"]["nearest_photo_id"] not in {"p01"}
    assert by_id["p01"]["predicted_species"] == "deer"
    assert by_id["p09"]["predicted_species"] == "fox"
    assert by_id["p10"]["predicted_species"] == "deer"
    # Equal-weight mean over 6 groups: four at 1.0, two singleton at 0.0.
    assert body["overall_accuracy"] == pytest.approx(4 / 6)
    assert [g["size"] for g in body["groups"]] == [2, 2, 2, 2, 1, 1]


def test_single_group_is_not_evaluable(client):
    payload = demo_payload("one-group")
    payload["pairs"] = [
        {"photo_a": f"p{i:02d}", "photo_b": f"p{i + 1:02d}"} for i in range(1, 10)
    ]
    response = client.post("/batches", json=payload)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["group_count"] == 1
    assert body["evaluable"] is False
    assert body["overall_accuracy"] is None
    assert body["predictions"] == []
    assert "single group" in body["message"]
    assert body["groups"][0]["accuracy"] is None


def test_validation_rejects_bad_input(client):
    payload = demo_payload()
    payload["photos"][0]["features"] = [0] * 31
    assert client.post("/batches", json=payload).status_code == 422

    payload = demo_payload()
    payload["photos"][0]["features"][0] = 40000
    assert client.post("/batches", json=payload).status_code == 422

    payload = demo_payload()
    payload["photos"][0]["features"][0] = 1.5
    assert client.post("/batches", json=payload).status_code == 422

    payload = demo_payload()
    payload["pairs"].append({"photo_a": "p01", "photo_b": "missing"})
    assert client.post("/batches", json=payload).status_code == 422

    payload = demo_payload()
    payload["pairs"].append({"photo_a": "p01", "photo_b": "p02"})
    assert client.post("/batches", json=payload).status_code == 422


def test_batch_limit_is_1000(client):
    base = demo_payload()
    photo = base["photos"][0]
    payload = {
        "batch_id": "too-many",
        "photos": [
            {**photo, "photo_id": f"x{i:04d}", "features": [0] * 32} for i in range(1001)
        ],
        "pairs": [],
    }
    assert client.post("/batches", json=payload).status_code == 422

    payload["batch_id"] = "exactly-1000"
    payload["photos"] = payload["photos"][:1000]
    response = client.post("/batches", json=payload)
    assert response.status_code == 201, response.text
    assert response.json()["photo_count"] == 1000


def test_duplicate_batch_id_conflicts(client):
    assert client.post("/batches", json=demo_payload("dup")).status_code == 201
    assert client.post("/batches", json=demo_payload("dup")).status_code == 409


def test_correction_recomputes_only_changed_membership_photos(client):
    client.post("/batches", json=demo_payload("c1"))
    response = client.post(
        "/batches/c1/corrections",
        json={
            "expected_version": 1,
            "action": "add",
            "photo_a": "p01",
            "photo_b": "p03",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["version"] == 2
    assert body["recomputed_photo_ids"] == ["p01", "p02", "p03", "p04"]
    assert body["overall_accuracy"] == pytest.approx(2 / 5)

    evaluation = client.get("/batches/c1/evaluation").json()
    assert evaluation["version"] == 2
    assert evaluation["group_count"] == 5
    assert evaluation["groups"][0] == {
        "photo_ids": ["p01", "p02", "p03", "p04"],
        "size": 4,
        "correct": 0,
        "accuracy": 0.0,
    }
    by_id = {p["photo_id"]: p for p in evaluation["predictions"]}
    assert by_id["p01"]["nearest_photo_id"] == "p10"


def test_cycle_edge_inside_component_recomputes_nobody(client):
    payload = demo_payload("c2")
    payload["pairs"] = [
        {"photo_a": "p01", "photo_b": "p02"},
        {"photo_a": "p02", "photo_b": "p03"},
    ]
    assert client.post("/batches", json=payload).status_code == 201
    response = client.post(
        "/batches/c2/corrections",
        json={
            "expected_version": 1,
            "action": "add",
            "photo_a": "p01",
            "photo_b": "p03",
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["recomputed_photo_ids"] == []
    assert response.json()["version"] == 2
    evaluation = client.get("/batches/c2/evaluation").json()
    assert evaluation["groups"][0] == {
        "photo_ids": ["p01", "p02", "p03"],
        "size": 3,
        "correct": 3,
        "accuracy": pytest.approx(1.0),
    }


def test_stale_version_is_rejected_and_keeps_old_version(client):
    client.post("/batches", json=demo_payload("c3"))
    good = client.post(
        "/batches/c3/corrections",
        json={"expected_version": 1, "action": "remove", "photo_a": "p01", "photo_b": "p02"},
    )
    assert good.status_code == 200
    before = client.get("/batches/c3/evaluation").json()

    stale = client.post(
        "/batches/c3/corrections",
        json={"expected_version": 1, "action": "remove", "photo_a": "p05", "photo_b": "p06"},
    )
    assert stale.status_code == 409
    assert stale.json()["error"] == "version_conflict"
    after = client.get("/batches/c3/evaluation").json()
    assert after == before


def test_correction_errors(client):
    client.post("/batches", json=demo_payload("c4"))
    cases = [
        ({"expected_version": 1, "action": "add", "photo_a": "p01", "photo_b": "p02"}, 409),
        ({"expected_version": 1, "action": "remove", "photo_a": "p01", "photo_b": "p03"}, 409),
        ({"expected_version": 1, "action": "add", "photo_a": "p01", "photo_b": "zzz"}, 422),
        ({"expected_version": 1, "action": "add", "photo_a": "p01", "photo_b": "p01"}, 422),
        ({"action": "add", "photo_a": "p01", "photo_b": "p03"}, 422),
    ]
    for payload, status in cases:
        assert client.post("/batches/c4/corrections", json=payload).status_code == status
    assert client.get("/batches/nope/evaluation").status_code == 404


def test_split_into_single_group_disables_evaluation(client):
    payload = demo_payload("c5")
    payload["pairs"] = [{"photo_a": f"p{i:02d}", "photo_b": f"p{i + 1:02d}"} for i in range(1, 9)]
    payload["pairs"].append({"photo_a": "p10", "photo_b": "p06"})
    created = client.post("/batches", json=payload).json()
    assert created["evaluable"] is False
    response = client.post(
        "/batches/c5/corrections",
        json={"expected_version": 1, "action": "remove", "photo_a": "p10", "photo_b": "p06"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["evaluable"] is True, body
    # The whole batch was one group, so splitting it changes every photo's
    # same-group membership even though only p10's group actually shrank.
    assert body["recomputed_photo_ids"] == [f"p{i:02d}" for i in range(1, 11)]
    evaluation = client.get("/batches/c5/evaluation").json()
    assert evaluation["evaluable"] is True
    assert len(evaluation["predictions"]) == 10


def test_concurrent_corrections_only_one_wins(client):
    client.post("/batches", json=demo_payload("race"))
    results: list[int] = []
    lock = threading.Lock()

    def attempt(a: str, b: str) -> None:
        response = client.post(
            "/batches/race/corrections",
            json={"expected_version": 1, "action": "add", "photo_a": a, "photo_b": b},
        )
        with lock:
            results.append(response.status_code)

    threads = [
        threading.Thread(target=attempt, args=("p01", "p03")),
        threading.Thread(target=attempt, args=("p05", "p07")),
        threading.Thread(target=attempt, args=("p09", "p10")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results) == [200, 409, 409]
    evaluation = client.get("/batches/race/evaluation").json()
    assert evaluation["version"] == 2


def test_tie_break_uses_smallest_photo_id():
    # Four singleton groups: every target has the same two equidistant
    # candidates; the smallest photo ID must win.
    ids = ["a", "b", "c"]
    features = np.asarray([[0, 0], [10, 0], [10, 0]], dtype=np.int64)
    species = {"a": "x", "b": "y", "c": "z"}
    mapping = {"a": 0, "b": 1, "c": 2}
    # Target a at (0,0): b and c are both distance 10 -> b.
    result = predict_nearest_outside(ids, species, features, mapping, ["a"])
    assert result["a"]["nearest_photo_id"] == "b"
    # Target b at (10,0): a is nearest (10 vs 0) -- c is same-group? no,
    # each is its own group, so c at distance 0 wins.
    result = predict_nearest_outside(ids, species, features, mapping, ["b"])
    assert result["b"]["nearest_photo_id"] == "c"


def test_tie_break_exact_tie_across_groups():
    ids = ["a", "b", "c", "d"]
    features = np.asarray([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.int64)
    species = {i: i for i in ids}
    mapping = {"a": 0, "b": 0, "c": 1, "d": 2}
    result = predict_nearest_outside(ids, species, features, mapping, ["a"])
    # Outside a's group: c at 10 and d at 14.14 -> c.
    assert result["a"]["nearest_photo_id"] == "c"


def test_grouping_and_membership_change_detection():
    groups = build_groups(["p2", "p1", "p3"], [("p1", "p2")])
    assert groups == [["p1", "p2"], ["p3"]]
    assert changed_membership_ids(groups, groups, ["p1", "p2", "p3"]) == []
    split = build_groups(["p2", "p1", "p3"], [])
    assert changed_membership_ids(groups, split, ["p1", "p2", "p3"]) == ["p1", "p2"]


def test_reads_see_previous_version_while_recompute_is_in_flight(client, monkeypatch):
    import time
    import app.service as service

    original = service.predict_nearest_outside

    def slow_predict(*args, **kwargs):
        time.sleep(0.6)
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "predict_nearest_outside", slow_predict)
    client.post("/batches", json=demo_payload("inflight"))
    observed: list[dict] = []

    def correct() -> None:
        response = client.post(
            "/batches/inflight/corrections",
            json={"expected_version": 1, "action": "add", "photo_a": "p01", "photo_b": "p03"},
        )
        observed.append({"status": response.status_code, "version": response.json()["version"]})

    thread = threading.Thread(target=correct)
    thread.start()
    time.sleep(0.2)
    during = client.get("/batches/inflight/evaluation")
    thread.join()

    assert during.status_code == 200
    assert during.json()["version"] == 1
    assert observed == [{"status": 200, "version": 2}]
    assert client.get("/batches/inflight/evaluation").json()["version"] == 2


def test_failed_correction_rolls_back_completely(client, monkeypatch):
    import app.service as service

    client.post("/batches", json=demo_payload("rollback"))
    before = client.get("/batches/rollback/evaluation").json()

    def explode(*args, **kwargs):
        raise RuntimeError("simulated storage failure")

    monkeypatch.setattr(service, "_upsert_predictions", explode)
    with pytest.raises(RuntimeError):
        client.post(
            "/batches/rollback/corrections",
            json={"expected_version": 1, "action": "add", "photo_a": "p01", "photo_b": "p03"},
        )

    after = client.get("/batches/rollback/evaluation").json()
    assert after == before
    assert after["version"] == 1


def test_predictions_match_brute_force_on_random_batches():
    import random

    rng = random.Random(7)
    for _ in range(40):
        size = rng.randint(2, 9)
        ids = sorted(f"r{i:02d}" for i in range(size))
        features = np.asarray(
            [[rng.randint(-5, 5), rng.randint(-5, 5)] for _ in ids], dtype=np.int64
        )
        species = {photo_id: rng.choice(["a", "b", "c"]) for photo_id in ids}
        labels = [rng.randint(0, 2) for _ in ids]  # deliberately not connected components
        mapping = dict(zip(ids, labels))
        result = predict_nearest_outside(ids, species, features, mapping, ids)
        if len(set(labels)) < 2:
            assert result == {}
            continue
        for photo_id in ids:
            best = None
            for other in ids:
                if mapping[other] == mapping[photo_id]:
                    continue
                squared = int(
                    ((features[ids.index(photo_id)] - features[ids.index(other)]) ** 2).sum()
                )
                candidate = (squared, other)
                if best is None or candidate < best:
                    best = candidate
            assert result[photo_id]["nearest_photo_id"] == best[1]
            assert result[photo_id]["distance"] == pytest.approx(best[0] ** 0.5)
            assert result[photo_id]["predicted_species"] == species[best[1]]
