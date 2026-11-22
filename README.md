# Wildlife Camera Evaluation API

Same-session camera photos are near-duplicates of each other. Scoring them
with a plain leave-one-out nearest neighbour would let each photo "recognise"
its own burst and inflate accuracy. This service treats directly or indirectly
linked photos as one group and scores every photo only against photos
**outside** its group.

Stack: FastAPI, NumPy, SQLite. A batch holds 1-1000 photos; each photo has a
client-supplied ID, a known species, and 32 signed 16-bit integer features.
After a batch exists, the only mutation is adding/removing one photo link.

## Run with Docker Compose

```bash
docker compose up --build
# docs: http://127.0.0.1:8000/docs
```

The SQLite file lives on the `eval-data` volume at
`/data/evaluations.sqlite3`. Override with the `EVALUATION_DB` env var.

Without Docker:

```bash
pip install -r requirements.txt
EVALUATION_DB=data/evaluations.sqlite3 uvicorn app.main:app --port 8000
```

## API

### `POST /batches` → 201

```json
{
  "batch_id": "demo-batch",          // optional; generated when omitted
  "photos": [{"photo_id": "p01", "species": "deer", "features": [0, 0, "... 32 int16 values"]}],
  "pairs":  [{"photo_a": "p01", "photo_b": "p02"}]
}
```

Returns the version-1 evaluation. Validation (all 422): 1-1000 photos,
unique photo IDs, exactly 32 strict integers in `[-32768, 32767]`, pair
endpoints must exist in the batch, duplicate/self pairs rejected. Reusing a
`batch_id` returns 409.

### `GET /batches/{batch_id}/evaluation` → 200

Current committed version: `version`, `groups`, every photo's prediction,
per-group accuracy, and `overall_accuracy` (equal-weight mean of group
accuracies). With a single group the response is `evaluable: false`,
`overall_accuracy: null`, `predictions: []`, and an explanatory `message`.

### `POST /batches/{batch_id}/corrections` → 200

```json
{"expected_version": 1, "action": "add", "photo_a": "p01", "photo_b": "p03"}
```

- `expected_version` is the reviewer's current version; a stale value returns
  **409 `version_conflict`** and the old version stays fully intact.
- `add` of an existing link / `remove` of a missing link returns 409;
  unknown endpoints return 422.
- Only photos whose same-group membership changed are re-predicted; the list
  is returned as `recomputed_photo_ids`. Adding a cycle edge inside one
  component changes nobody (empty list). Ending up with one group flips the
  batch to `evaluable: false` and drops predictions.
- Grouping, predictions, and the overall score switch versions atomically in
  one `BEGIN IMMEDIATE` transaction. Concurrent corrections against the same
  version: exactly one commits, the rest get 409.
- Reads use SQLite WAL snapshots, so `GET .../evaluation` keeps serving the
  old version while a correction is being computed and switches only on commit.

Predictions: nearest Euclidean distance over photos outside the group, using
exact integer squared distances; ties are broken by the smallest photo ID.

## Offline reference packages (`POST /batches/{batch_id}/offline-package`)

A patrol tablet has room for only a few reference photos. Given 1-16
**candidate references** and 1-40 additional **verification photos**, this
endpoint selects the smallest subset of candidates that reproduces the
full-library verdict on every verification photo under the same
nearest-outside-group rule, both with every camera group enabled and once
per candidate camera group while that group is simulated as disabled:

```json
{
  "expected_version": 1,
  "candidate_photo_ids": ["r1", "r2", "r3", "r4", "r5"],
  "verification_photo_ids": ["v1", "v2"],
  "capacity": 3
}
```

For every (verification photo, scenario) the package verdict and the full
candidate-library verdict must name the same species, or both must have no
usable reference ("no reference": every eligible candidate is in the
photo's own group or in the disabled group). The solver enumerates all
2**candidate subsets exactly (at most 65536), ranks candidates by
`(squared distance, photo ID)`, and returns the fewest photos; on equal
size it returns the lexicographically smallest tuple of sorted photo IDs.
Responses include, per packaged photo, its ID, 32 features, species and
**`group_id`** — the smallest candidate reference ID in its camera group.

- When the minimum package needs more than `capacity` photos the endpoint
  returns **409 `capacity_exceeded`** with `details.required_size`, so the
  tablet can report how many slots it would have needed.
- A stale `expected_version` returns **409 `version_conflict`**; unknown
  candidate/verification IDs return 422, as do candidate/verification
  overlaps, empty ID lists, counts outside 1-16 / 1-40, or `capacity < 1`.
- The computation takes a read-only WAL snapshot and never holds a write
  lock, so reviewers can still commit corrections while a package is being
  solved. Before answering, the same-group relationships **between the
  selected photos only** are re-checked against the latest committed
  version. If a correction changed them, the call returns **409
  `group_relationship_changed`** with the current version; otherwise the
  answer is stamped with the re-check version (which may be newer than
  `expected_version` when an unrelated correction committed in between).
- A scenario with no full-library reference imposes no requirement: any
  subset of the same candidates under the same eligibility also has no
  reference, so both sides agree on "no reference". When *every* scenario
  is like that the minimum package is the empty list.

### `POST /batches/{batch_id}/offline-package` → 200

Returns `version` (re-check version), `package_size`, the selected
`package_photos`, every candidate with its group label under
`candidate_groups`, and `scenarios` — one entry per verification photo per
outage (`all_enabled`, or `group_disabled` with `disabled_group_id`)
containing the full-library and packaged nearest photo IDs and species
(`null` means "no reference").

### Demonstration: how disabling a camera group changes the package

```bash
BASE_URL=http://127.0.0.1:8000 bash examples/offline_requests.sh
```

The demo batch has a deer group (`r1,r2`), a fox group (`r3,r4`) and a
singleton otter (`r5`). With all groups enabled, `v1` matches the deer
`r1`; once the deer camera group is simulated as disabled its nearest
surviving reference is the fox `r3`, so the package must carry `r3` as
well. Disabling the otter group makes `v2` fall back to `r1`. The smallest
surviving package is therefore `r1 + r3 + r5`: asking for capacity 2
returns `capacity_exceeded` with `required_size: 3`, and capacity 3
returns the package plus every side-by-side verdict. The script also shows
a correction committing mid-flight and a stale version-1 retry being
rejected.


## Demonstration (before/after a correction)

```bash
BASE_URL=http://127.0.0.1:8000 bash examples/requests.sh
```

`examples/create_batch.json` has 10 photos in 6 groups (two same-session pairs
at each of two sites plus two singletons). Version 1 scores
`4/6 = 0.6667`; adding `p01-p03` merges two deer pairs into one 4-photo group,
recomputes exactly `p01,p02,p03,p04`, and the score drops to `2/5 = 0.4`.
The script also shows a stale version-1 correction rejected with 409.
`examples/requests.sh` writes the raw responses next to the payloads.

## Tests

```bash
pip install -r requirements-dev.txt
python3 -m pytest -q
```

18 tests cover ingestion/validation, the 1000-photo limit, grouping, exact tie
breaking, brute-force cross-checks, incremental recomputation, version
conflicts, concurrent writers, rollback on failure, and reads during an
in-flight correction.

`tests/test_offline_package.py` adds 14 tests for the package API: minimum
selection across group outages, group labels, capacity shortfalls, stale
versions, unknown photos, request bounds and overlaps, mutual "no
reference", lexicographic tie-breaks, re-check conflicts when a correction
changes selected-photo relationships, version stamping after unrelated
corrections, corrections committing while a package is solving, an
independent brute-force oracle over random batches, and a timed 16
candidate × 40 verification run.
