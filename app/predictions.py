"""Nearest-outside-group prediction and equal-weight group scoring."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def predict_nearest_outside(
    photo_ids: Sequence[str],
    species_by_id: dict[str, str],
    features: np.ndarray,
    group_by_id: dict[str, int],
    target_photo_ids: Sequence[str],
) -> dict[str, dict]:
    """Predict species for only the requested photos.

    For each target, consider every photo outside its own group. The nearest
    photo is chosen by exact squared Euclidean distance (integer arithmetic),
    and an exact tie is broken by the lexicographically smallest photo ID.
    `photo_ids` must be sorted ascending so array indices respect ID order.

    Returns an empty mapping when fewer than two groups exist: with no photo
    outside the sole group, prediction is impossible.
    """
    group_count = len(set(group_by_id.values()))
    targets = sorted(set(target_photo_ids))
    if list(photo_ids) != sorted(photo_ids):
        raise ValueError("photo_ids must be sorted ascending for deterministic ties")
    if group_count < 2 or not targets:
        return {}

    id_to_index = {photo_id: index for index, photo_id in enumerate(photo_ids)}
    target_indices = np.asarray([id_to_index[photo_id] for photo_id in targets], dtype=np.int64)
    group_array = np.asarray([group_by_id[photo_id] for photo_id in photo_ids], dtype=np.int64)

    matrix = np.asarray(features, dtype=np.int64)
    norms = np.einsum("ij,ij->i", matrix, matrix)
    # Exact integer squared distances: 32 * (65535 ** 2) is far below int64 max.
    squared = (
        norms[target_indices, None]
        + norms[None, :]
        - 2 * (matrix[target_indices] @ matrix.T)
    )
    outside = group_array[None, :] != group_array[target_indices, None]

    predictions: dict[str, dict] = {}
    unreachable = np.iinfo(np.int64).max
    for row, target_index in enumerate(target_indices):
        candidates = np.where(outside[row], squared[row], unreachable)
        minimum = int(candidates.min())
        if minimum == unreachable:  # Defensive; >=2 groups always has candidates.
            continue
        # `photo_ids` is sorted, so the first flat index among equal distances
        # is the lexicographically smallest photo ID.
        tie_indices = np.flatnonzero(outside[row] & (squared[row] == minimum))
        nearest_index = int(tie_indices[0])
        target_id = photo_ids[int(target_index)]
        nearest_id = photo_ids[nearest_index]
        predicted_species = species_by_id[nearest_id]
        predictions[target_id] = {
            "photo_id": target_id,
            "known_species": species_by_id[target_id],
            "predicted_species": predicted_species,
            "nearest_photo_id": nearest_id,
            "distance": math.sqrt(minimum),
            "correct": predicted_species == species_by_id[target_id],
        }
    return predictions


def summarize_groups(
    groups: Sequence[Sequence[str]],
    predictions_by_id: dict[str, dict],
    evaluable: bool,
) -> tuple[list[dict], float | None]:
    """Compute per-group accuracy and their equal-weight mean."""
    group_output: list[dict] = []
    accuracies: list[float] = []
    for members in groups:
        if evaluable:
            missing = [photo_id for photo_id in members if photo_id not in predictions_by_id]
            if missing:
                raise RuntimeError(f"missing predictions for photos: {missing}")
            correct = sum(1 for photo_id in members if predictions_by_id[photo_id]["correct"])
            accuracy = correct / len(members)
            accuracies.append(accuracy)
        else:
            correct = None
            accuracy = None
        group_output.append(
            {
                "photo_ids": list(members),
                "size": len(members),
                "correct": correct,
                "accuracy": accuracy,
            }
        )

    overall = sum(accuracies) / len(accuracies) if evaluable else None
    return group_output, overall
