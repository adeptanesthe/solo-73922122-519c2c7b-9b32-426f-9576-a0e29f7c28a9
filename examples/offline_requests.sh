#!/usr/bin/env bash
# Demonstrates the offline reference-package API and, in particular, how
# simulating the outage of one candidate camera group changes which photos
# have to be packaged. Run against `docker compose up --build`.
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH="patrol-demo"

echo "== 1. Create the patrol batch (version 1) =="
echo "Candidate groups: r1+r2 (deer), r3+r4 (fox), r5 singleton (otter)."
echo "v1 sits next to r1; v2 sits next to r5."
curl -sS -X POST "$BASE_URL/batches" \
  -H 'Content-Type: application/json' \
  --data @"$HERE/offline_batch.json" \
  | python3 -m json.tool > "$HERE/offline_batch_response.json"
python3 - "$HERE/offline_batch_response.json" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
print("version:", body["version"], "groups:", body["group_count"],
      "photos:", body["photo_count"])
PY

echo
echo "== 2. Ask for a package with capacity 2 -> 409 capacity_exceeded =="
curl -sS -o "$HERE/offline_capacity2_response.json" -w 'HTTP %{http_code}\n' \
  -X POST "$BASE_URL/batches/$BATCH/offline-package" \
  -H 'Content-Type: application/json' \
  --data @"$HERE/offline_package_capacity2.json"
python3 - "$HERE/offline_capacity2_response.json" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
print("error:", body["error"], "| details:", body["details"])
PY

echo
echo "== 3. Ask again with capacity 3 -> the minimum package =="
curl -sS -X POST "$BASE_URL/batches/$BATCH/offline-package" \
  -H 'Content-Type: application/json' \
  --data @"$HERE/offline_package_capacity3.json" \
  | python3 -m json.tool > "$HERE/offline_package_response.json"
python3 - "$HERE/offline_package_response.json" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
print("answered at version:", body["version"], "| package size:",
      body["package_size"], "/ capacity:", body["capacity"])
print("packaged photos:", [p["photo_id"] for p in body["package_photos"]])
for photo in body["package_photos"]:
    print(f'  {photo["photo_id"]}: species={photo["species"]}, '
          f'group_id={photo["group_id"]} (smallest candidate ID in its group)')

print()
print("Per-verification verdicts (full library vs. selected package):")
for scenario in body["scenarios"]:
    target = scenario["disabled_group_id"] or "all groups enabled"
    print(f'  {scenario["verification_photo_id"]} | outage: {target:>18} | '
          f'full={str(scenario["full_species"]):>5} '
          f'({scenario["full_nearest_photo_id"]}) | '
          f'package={str(scenario["package_species"]):>5} '
          f'({scenario["package_nearest_photo_id"]})')
PY

echo
echo "== 4. Why does disabling a group grow the package? =="
python3 - "$HERE/offline_package_response.json" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
rows = {(s["verification_photo_id"], s["disabled_group_id"]): s
        for s in body["scenarios"]}
v1_deer_off = rows[("v1", "r1")]
v2_otter_off = rows[("v2", "r5")]
print("With every group enabled, v1's nearest reference is r1 (deer).")
print("When the deer camera group r1 is disabled, v1's nearest surviving")
print("reference becomes", v1_deer_off["full_nearest_photo_id"],
      f'({v1_deer_off["full_species"]}); the package therefore needs r3 too.')
print("When the otter group r5 is disabled, v2 falls back to",
      v2_otter_off["full_nearest_photo_id"],
      f'({v2_otter_off["full_species"]}), so r1 must stay in the package.')
print("Minimum surviving set: r1 + r3 + r5.")
PY

echo
echo "== 5. A correction commits while packages are solvable =="
curl -sS -X POST "$BASE_URL/batches/$BATCH/corrections" \
  -H 'Content-Type: application/json' \
  --data @"$HERE/offline_correction.json" \
  | python3 -m json.tool > "$HERE/offline_correction_response.json"
python3 - "$HERE/offline_correction_response.json" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
print("batch is now version:", body["version"])
PY

echo
echo "== 6. Retry the package against version 1 -> 409 version_conflict =="
curl -sS -o "$HERE/offline_stale_response.json" -w 'HTTP %{http_code}\n' \
  -X POST "$BASE_URL/batches/$BATCH/offline-package" \
  -H 'Content-Type: application/json' \
  --data @"$HERE/offline_package_capacity3.json"
cat "$HERE/offline_stale_response.json"; echo
