"""Connected-component grouping and membership-change detection."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def normalize_pair(photo_a: str, photo_b: str) -> tuple[str, str]:
    """Photo pairs are undirected; store them in a deterministic order."""
    return (photo_a, photo_b) if photo_a <= photo_b else (photo_b, photo_a)


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def build_groups(
    photo_ids: Sequence[str], pairs: Iterable[tuple[str, str]]
) -> list[list[str]]:
    """Return connected components.

    Each group is sorted by photo ID, and groups themselves are sorted by their
    smallest photo ID so API output is deterministic.
    """
    union_find = UnionFind()
    for photo_id in photo_ids:
        union_find.add(photo_id)
    for photo_a, photo_b in pairs:
        union_find.add(photo_a)
        union_find.add(photo_b)
        union_find.union(photo_a, photo_b)

    components: dict[str, list[str]] = {}
    for photo_id in photo_ids:
        root = union_find.find(photo_id)
        components.setdefault(root, []).append(photo_id)

    groups = [sorted(members) for members in components.values()]
    groups.sort(key=lambda members: members[0])
    return groups


def group_index(groups: Sequence[Sequence[str]]) -> dict[str, int]:
    """Map every photo ID to a stable index among the supplied groups."""
    mapping: dict[str, int] = {}
    for index, members in enumerate(groups):
        for photo_id in members:
            mapping[photo_id] = index
    return mapping


def changed_membership_ids(
    old_groups: Sequence[Sequence[str]],
    new_groups: Sequence[Sequence[str]],
    photo_ids: Sequence[str],
) -> list[str]:
    """Return IDs whose exact set of same-group photos changed.

    Membership is compared as a set of co-members, not as an arbitrary group
    label. Therefore adding a cycle edge inside one component correctly yields
    no changed photos, while splitting a component marks every photo in the
    old component.
    """
    old_signature: dict[str, frozenset[str]] = {}
    for members in old_groups:
        signature = frozenset(members)
        for photo_id in members:
            old_signature[photo_id] = signature

    new_signature: dict[str, frozenset[str]] = {}
    for members in new_groups:
        signature = frozenset(members)
        for photo_id in members:
            new_signature[photo_id] = signature

    return sorted(
        photo_id
        for photo_id in photo_ids
        if old_signature.get(photo_id) != new_signature.get(photo_id)
    )
