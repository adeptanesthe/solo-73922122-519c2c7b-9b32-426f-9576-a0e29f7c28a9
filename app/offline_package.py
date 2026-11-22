"""Minimum offline reference-package selection.

A package is a subset of the candidate reference photos. It must reproduce,
for every verification photo and every tolerated camera-group outage, the
verdict that the *full* candidate library gives under the original
"nearest neighbour outside the photo's own group" rule:

* both verdicts name the same species, or
* both have no usable reference at all ("no reference").

The tolerated outages are: every camera group enabled, plus one outage per
candidate camera group. With at most 16 candidates, all 2**16 = 65536
subsets can be enumerated exactly. For every (verification photo, scenario)
we precompute the full-library nearest candidate, rank every candidate by
(squared distance, ID), and walk the ranks over all subsets in one
vectorized pass to find each package's nearest reference.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np

# Sentinel group index for the scenario in which no candidate group is off.
NO_DISABLED = -1

# Larger than any real squared distance: 32 * 65535**2 stays below int64 max.
UNREACHABLE = np.iinfo(np.int64).max


def squared_distance_matrix(features: np.ndarray) -> np.ndarray:
    """Exact integer squared Euclidean distances (int64)."""
    matrix = np.asarray(features, dtype=np.int64)
    norms = np.einsum("ij,ij->i", matrix, matrix)
    squared = norms[:, None] + norms[None, :] - 2 * (matrix @ matrix.T)
    return squared.astype(np.int64, copy=False)


def nearest_candidate(distances: np.ndarray, eligible: np.ndarray) -> int | None:
    """Smallest-ID eligible candidate at the minimum squared distance.

    Candidates are indexed in ascending photo-ID order, so the first index
    among equal distances is the lexicographically smallest photo ID.
    """
    if not bool(eligible.any()):
        return None
    values = np.where(eligible, distances, UNREACHABLE)
    minimum = int(values.min())
    return int(np.flatnonzero(eligible & (distances == minimum))[0])


def solve_minimum_package(
    candidate_ids: Sequence[str],
    verification_ids: Sequence[str],
    candidate_group: Sequence[int],
    verification_group: Sequence[int],
    candidate_species: Sequence[str],
    distances: np.ndarray,
    candidate_group_indices: Sequence[int],
) -> tuple[list[int], list[dict]]:
    """Return selected candidate indices and per-scenario verdicts.

    Selection rule: fewest photos; on equal size, the lexicographically
    smallest tuple of sorted photo IDs. ``candidate_ids`` must be sorted
    ascending, so numeric subset order and ``itertools.combinations`` respect
    the ID-lexicographic tie-break.

    Scenarios are, in order: all groups enabled (``NO_DISABLED``), then one
    scenario per candidate group (its index).
    """
    candidate_count = len(candidate_ids)
    candidate_group_array = np.asarray(candidate_group, dtype=np.int64)
    candidate_species_array = np.asarray(candidate_species)
    disabled_values = [NO_DISABLED, *candidate_group_indices]
    subset_count = 1 << candidate_count
    subset_indices = np.arange(subset_count, dtype=np.int64)

    # One constraint per (verification photo, scenario). A constraint without
    # a full-library reference imposes nothing: the package is a subset of
    # the same candidates under the same eligibility, so it has no reference
    # either and both sides answer "no reference".
    constraints: list[dict] = []
    verdict_template: list[dict] = []
    for v_index, verification_id in enumerate(verification_ids):
        own_group = verification_group[v_index]
        row = distances[v_index]
        for disabled in disabled_values:
            eligible = (candidate_group_array != own_group) & (
                candidate_group_array != disabled
            )
            full_index = nearest_candidate(row, eligible)
            verdict_template.append(
                {
                    "disabled_group": disabled,
                    "verification_photo_id": verification_id,
                    "full_index": full_index,
                }
            )
            if full_index is None:
                continue
            target_species = candidate_species_array[full_index]
            required = eligible & (candidate_species_array == target_species)
            constraints.append(
                {
                    "eligible": eligible,
                    "required": required,
                    "row": row,
                }
            )

    constraint_count = len(constraints)
    valid_subsets = (1 << subset_count) - 1
    if constraint_count:
        eligibility_matrix = np.column_stack(
            [constraint["eligible"] for constraint in constraints]
        )
        distance_matrix = np.column_stack(
            [constraint["row"] for constraint in constraints]
        )
        required_matrix = np.column_stack(
            [constraint["required"] for constraint in constraints]
        )
        # Rank candidates per constraint: rank 0 is the best (distance, ID).
        # lexsort orders by the last key first; both keys share the matrix
        # shape, so ties on distance resolve to the smallest candidate index.
        candidate_index = np.broadcast_to(
            np.arange(candidate_count)[:, None], distance_matrix.shape
        )
        order = np.lexsort((candidate_index, distance_matrix), axis=0)
        columns = np.arange(constraint_count)

        # Walk ranks best->worst and assign each still-unassigned subset the
        # first candidate it contains that is eligible for the constraint.
        nearest = np.full((subset_count, constraint_count), -1, dtype=np.int16)
        for rank in range(candidate_count):
            candidate = order[rank]
            membership = (subset_indices[:, None] >> candidate[None, :]) & 1
            usable = membership.astype(bool) & eligibility_matrix[
                candidate[None, :], columns[None, :]
            ]
            fill = (nearest < 0) & usable
            nearest[fill] = np.broadcast_to(
                candidate[None, :], nearest.shape
            )[fill]

        # A subset is valid when every constraint's nearest packaged
        # candidate carries the full-library target species.
        chosen = np.maximum(nearest, 0)
        required_hit = np.take_along_axis(required_matrix, chosen, axis=0)
        required_hit &= nearest >= 0
        valid_flags = required_hit.all(axis=1)
        valid_subsets = 0
        for subset in np.flatnonzero(valid_flags).tolist():
            valid_subsets |= 1 << int(subset)

    # Numeric subset order orders same-size tuples lexicographically (bit n
    # is candidate n), so the first hit per size is the tie-break winner.
    selected: list[int] = []
    found = False
    for size in range(candidate_count + 1):
        for combo in combinations(range(candidate_count), size):
            subset_bits = sum(1 << index for index in combo)
            if (valid_subsets >> subset_bits) & 1:
                selected = list(combo)
                found = True
                break
        if found:
            break
    if not found:  # Defensive: the full candidate set is always valid.
        raise RuntimeError("no valid offline package exists")

    selected_set = set(selected)
    verdicts: list[dict] = []
    for template in verdict_template:
        v_index = verification_ids.index(template["verification_photo_id"])
        eligible = (candidate_group_array != verification_group[v_index]) & (
            candidate_group_array != template["disabled_group"]
        )
        package_eligible = np.zeros(candidate_count, dtype=bool)
        package_eligible[list(selected_set)] = eligible[list(selected_set)]
        package_index = nearest_candidate(distances[v_index], package_eligible)
        verdicts.append(
            {
                "disabled_group": template["disabled_group"],
                "verification_photo_id": template["verification_photo_id"],
                "full_index": template["full_index"],
                "package_index": package_index,
            }
        )

    return selected, verdicts
