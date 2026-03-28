#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Simple tok/s benchmark for vllm-greenboost
#
# Sends a prompt to the local vLLM endpoint and measures generation speed.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

# Load config if available
if [[ -f "${REPO_DIR}/.env" ]]; then
    source "${REPO_DIR}/.env"
fi

HOST="${VLLM_HOST:-localhost}"
PORT="${VLLM_PORT:-8000}"
MAX_TOKENS="${1:-100}"
BASE_URL="http://${HOST}:${PORT}"

echo "vllm-greenboost benchmark"
echo "════════════════════════════════════════════════════════════════"
echo "Endpoint:   ${BASE_URL}"
echo "Max tokens: ${MAX_TOKENS}"
echo ""

# Wait for the API to be ready
echo "Checking API availability..."
for i in $(seq 1 10); do
    if curl -sf "${BASE_URL}/health" &>/dev/null || curl -sf "${BASE_URL}/v1/models" &>/dev/null; then
        break
    fi
    if [[ $i -eq 10 ]]; then
        echo "ERROR: vLLM API not responding at ${BASE_URL}"
        echo "       Is the service running?  systemctl status vllm"
        exit 1
    fi
    echo "  Waiting for API... (attempt ${i}/10)"
    sleep 5
done

# Discover model name
MODEL=$(curl -sf "${BASE_URL}/v1/models" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data['data'][0]['id'])
" 2>/dev/null || echo "${MODEL_PATH:-unknown}")

echo "Model:      ${MODEL}"
echo ""

PROMPT="Write a detailed explanation of how GPU memory management works in modern deep learning frameworks. Cover topics including memory allocation strategies, gradient checkpointing, and memory-efficient attention mechanisms."

echo "Sending prompt ($(echo -n "${PROMPT}" | wc -c) chars)..."
echo "────────────────────────────────────────────────────────────────"

START_TIME=$(date +%s%N)

RESPONSE=$(curl -sf "${BASE_URL}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "$(python3 -c "
import json
print(json.dumps({
    'model': '${MODEL}',
    'prompt': $(python3 -c "import json; print(json.dumps('${PROMPT}'))"),
    'max_tokens': ${MAX_TOKENS},
    'temperature': 0.7
}))
")")

END_TIME=$(date +%s%N)

if [[ -z "${RESPONSE}" ]]; then
    echo "ERROR: Empty response from API."
    exit 1
fi

ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))
COMPLETION_TOKENS=$(echo "${RESPONSE}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('usage', {}).get('completion_tokens', 0))
" 2>/dev/null || echo "0")

PROMPT_TOKENS=$(echo "${RESPONSE}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('usage', {}).get('prompt_tokens', 0))
" 2>/dev/null || echo "0")

TEXT=$(echo "${RESPONSE}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data['choices'][0]['text'][:200] + '...')
" 2>/dev/null || echo "(could not parse)")

echo ""
echo "Response preview:"
echo "  ${TEXT}"
echo ""
echo "────────────────────────────────────────────────────────────────"
echo "Results"
echo "────────────────────────────────────────────────────────────────"

if [[ "${COMPLETION_TOKENS}" -gt 0 && "${ELAPSED_MS}" -gt 0 ]]; then
    TOKS_PER_SEC=$(python3 -c "print(f'{${COMPLETION_TOKENS} / (${ELAPSED_MS} / 1000):.2f}')")
    echo "  Prompt tokens:     ${PROMPT_TOKENS}"
    echo "  Completion tokens: ${COMPLETION_TOKENS}"
    echo "  Wall time:         ${ELAPSED_MS} ms"
    echo "  Throughput:        ${TOKS_PER_SEC} tok/s"
else
    echo "  Wall time: ${ELAPSED_MS} ms"
    echo "  (Could not compute tok/s — check response)"
fi

echo ""
echo "For reference — expected throughput with GreenBoost Path A/B:"
echo "  PCIe 3.0 x16: ~0.5-1.0 tok/s (model-dependent)"
echo "  PCIe 4.0 x16: ~1.0-2.0 tok/s"
echo "  PCIe 5.0 x16: ~2.0-4.0 tok/s"
