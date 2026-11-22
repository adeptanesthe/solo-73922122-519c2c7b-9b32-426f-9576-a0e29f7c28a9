#!/usr/bin/env bash
# Demonstrates the version-1 evaluation, one reviewer correction, then the
# version-2 evaluation. Run against `docker compose up --build`.
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== 1. Create batch (version 1) =="
curl -sS -X POST "$BASE_URL/batches" \
  -H 'Content-Type: application/json' \
  --data @"$HERE/create_batch.json" \
  | python3 -m json.tool > "$HERE/v1_evaluation.json"
python3 - "$HERE/v1_evaluation.json" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
print("version:", body["version"], "groups:", body["group_count"],
      "evaluable:", body["evaluable"], "overall_accuracy:", body["overall_accuracy"])
for prediction in body["predictions"]:
    print(f'  {prediction["photo_id"]} {prediction["known_species"]:>5} -> '
          f'{prediction["predicted_species"]:>5} via {prediction["nearest_photo_id"]} '
          f'({"correct" if prediction["correct"] else "wrong"})')
PY

echo
echo "== 2. Reviewer adds link p01-p03 against version 1 =="
curl -sS -X POST "$BASE_URL/batches/demo-batch/corrections" \
  -H 'Content-Type: application/json' \
  --data @"$HERE/after_correction.json" \
  | python3 -m json.tool > "$HERE/correction_response.json"
python3 - "$HERE/correction_response.json" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
print("version:", body["previous_version"], "->", body["version"],
      "recomputed:", body["recomputed_photo_ids"],
      "overall_accuracy:", body["overall_accuracy"])
PY

echo
echo "== 3. Read current evaluation (version 2) =="
curl -sS "$BASE_URL/batches/demo-batch/evaluation" \
  | python3 -m json.tool > "$HERE/v2_evaluation.json"
python3 - "$HERE/v2_evaluation.json" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
print("version:", body["version"], "groups:", body["group_count"],
      "overall_accuracy:", body["overall_accuracy"])
for group in body["groups"]:
    print("  group", group["photo_ids"], "accuracy", group["accuracy"])
PY

echo
echo "== 4. A stale correction against version 1 is rejected =="
curl -sS -o "$HERE/stale_response.json" -w 'HTTP %{http_code}\n' \
  -X POST "$BASE_URL/batches/demo-batch/corrections" \
  -H 'Content-Type: application/json' \
  --data '{"expected_version": 1, "action": "remove", "photo_a": "p05", "photo_b": "p06"}'
cat "$HERE/stale_response.json"; echo
