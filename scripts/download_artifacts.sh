#!/usr/bin/env bash
# download_artifacts.sh — Download GitHub Actions artifacts for a given run.
#
# Usage:
#   ./scripts/download_artifacts.sh <run_id> [output_dir] [name_filter]
#
# Examples:
#   # Download all artifacts from run 26763854552
#   ./scripts/download_artifacts.sh 26763854552
#
#   # Download to a custom directory
#   ./scripts/download_artifacts.sh 26763854552 ~/Downloads/run_26763854552
#
#   # Download only artifacts matching a name pattern (Python re.search regex)
#   ./scripts/download_artifacts.sh 26763854552 ./artifacts "bmk_agentic|agentic_aggregated|results_bmk"
#
#   # Skip server_logs (large), download everything else
#   ./scripts/download_artifacts.sh 26763854552 ./artifacts "^(?!server_logs)"
#
# Requirements:
#   - gh CLI authenticated with an OAuth token (not a classic PAT).
#     If GITHUB_TOKEN env var is set to a classic PAT, unset it first:
#       unset GITHUB_TOKEN && ./scripts/download_artifacts.sh ...
#   - unzip

set -euo pipefail

# Python interpreter — uses conda inferencemax env
PYTHON="${PYTHON:-/c/Users/gguasti/AppData/Local/miniforge3/envs/inferencemax/python}"
if ! command -v "$PYTHON" &>/dev/null; then
    PYTHON=$(command -v python3 2>/dev/null || command -v python 2>/dev/null)
fi

REPO="ROCm/InferenceMAX_rocm"
RUN_ID="${1:?Usage: $0 <run_id> [output_dir] [name_filter]}"
OUT_DIR="${2:-./artifacts/${RUN_ID}}"
NAME_FILTER="${3:-}"  # Python re.search pattern; empty = all

mkdir -p "$OUT_DIR"

echo "==> Fetching artifact list for run ${RUN_ID} ..."

# Fetch up to 100 artifacts per page; GitHub Actions max is typically <200 per run.
artifacts_json=$(gh api \
    "repos/${REPO}/actions/runs/${RUN_ID}/artifacts?per_page=100" \
    --jq '[.artifacts[] | {id:.id, name:.name, size_mb: (.size_in_bytes/1048576 | floor), expired:.expired}]')

# Handle pagination if there are more than 100 artifacts
page=2
while true; do
    page_json=$(gh api \
        "repos/${REPO}/actions/runs/${RUN_ID}/artifacts?per_page=100&page=${page}" \
        --jq '[.artifacts[] | {id:.id, name:.name, size_mb: (.size_in_bytes/1048576 | floor), expired:.expired}]')
    count=$(echo "$page_json" | "$PYTHON" -c "import sys,json; print(len(json.load(sys.stdin)))")
    if [ "$count" -eq 0 ]; then
        break
    fi
    # Merge arrays
    artifacts_json=$(printf '%s\n%s' "$artifacts_json" "$page_json" | \
        "$PYTHON" -c "
import sys, json
lines = sys.stdin.read().strip().split('\n')
merged = []
for line in lines:
    if line.strip():
        merged.extend(json.loads(line))
print(json.dumps(merged))
")
    page=$((page + 1))
done

total_count=$(echo "$artifacts_json" | "$PYTHON" -c "import sys,json; print(len(json.load(sys.stdin)))")
echo "==> Found ${total_count} artifacts."

# Apply name filter if provided
if [ -n "$NAME_FILTER" ]; then
    artifacts_json=$(echo "$artifacts_json" | "$PYTHON" -c "
import sys, json, re
arts = json.load(sys.stdin)
pattern = sys.argv[1]
out = [a for a in arts if re.search(pattern, a['name'])]
print(json.dumps(out))
" "$NAME_FILTER")
    filtered_count=$(echo "$artifacts_json" | "$PYTHON" -c "import sys,json; print(len(json.load(sys.stdin)))")
    echo "==> Filter '${NAME_FILTER}' matches ${filtered_count}/${total_count} artifacts."
fi

# Print list
echo ""
echo "$artifacts_json" | "$PYTHON" -c "
import sys, json
arts = json.load(sys.stdin)
for a in arts:
    expired = ' [EXPIRED]' if a['expired'] else ''
    print(f\"  [{a['id']}] {a['name']}  ({a['size_mb']} MB){expired}\")
"
echo ""

# Download loop
downloaded=0
skipped=0
failed=0

while IFS=$'\t' read -r art_id art_name art_expired; do
    dest="${OUT_DIR}/${art_name}"
    zip_file="${dest}.zip"

    if [ "$art_expired" = "True" ]; then
        echo "  SKIP (expired): ${art_name}"
        skipped=$((skipped + 1))
        continue
    fi

    if [ -d "$dest" ]; then
        echo "  SKIP (already extracted): ${art_name}"
        skipped=$((skipped + 1))
        continue
    fi

    echo -n "  Downloading: ${art_name} ... "
    if gh api \
        "repos/${REPO}/actions/artifacts/${art_id}/zip" \
        --method GET \
        -H "Accept: application/vnd.github+json" \
        > "$zip_file" 2>/dev/null; then
        mkdir -p "$dest"
        unzip -q "$zip_file" -d "$dest" && rm -f "$zip_file"
        echo "OK -> ${dest}"
        downloaded=$((downloaded + 1))
    else
        echo "FAILED"
        rm -f "$zip_file"
        failed=$((failed + 1))
    fi
done < <(echo "$artifacts_json" | "$PYTHON" -c "
import sys, json
arts = json.load(sys.stdin)
for a in arts:
    print(f\"{a['id']}\t{a['name']}\t{a['expired']}\")
")

echo ""
echo "==> Done. Downloaded: ${downloaded}  Skipped: ${skipped}  Failed: ${failed}"
echo "==> Output: ${OUT_DIR}"
