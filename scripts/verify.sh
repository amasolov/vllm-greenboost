#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Post-install health check for vllm-greenboost
# Run after setup.sh to confirm everything is working correctly.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PASS=0
FAIL=0
WARN=0

check() {
    local label="$1"; shift
    if "$@" &>/dev/null; then
        echo "  [PASS] ${label}"
        (( PASS++ ))
    else
        echo "  [FAIL] ${label}"
        (( FAIL++ ))
    fi
}

warn() {
    local label="$1"
    echo "  [WARN] ${label}"
    (( WARN++ ))
}

echo "vllm-greenboost verification"
echo "════════════════════════════════════════════════════════════════"

# ── NVIDIA driver ─────────────────────────────────────────────────────────────
echo ""
echo "NVIDIA Driver"
echo "────────────────────────────────────────────────────────────────"

check "nvidia-smi is available" command -v nvidia-smi

if command -v nvidia-smi &>/dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    echo "  [INFO] Driver version: ${DRIVER_VER:-unknown}"

    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo "  [INFO] GPU: ${GPU_NAME:-unknown} (${GPU_VRAM:-?})"

    # Check it is NOT the open kernel module
    KERNEL_VER=$(uname -r)
    NVIDIA_KO=$(find /lib/modules/"${KERNEL_VER}" -name 'nvidia.ko*' 2>/dev/null | head -1)
    if [[ -n "${NVIDIA_KO}" ]]; then
        LICENSE=$(modinfo "${NVIDIA_KO}" 2>/dev/null | awk '/^license/ {print $2}')
        if [[ "${LICENSE}" == "GPL" ]]; then
            echo "  [FAIL] Driver is open kernel module (GPL) — cuMemHostRegister won't work on Turing"
            (( FAIL++ ))
        else
            echo "  [PASS] Driver is proprietary (license: ${LICENSE:-NVIDIA})"
            (( PASS++ ))
        fi
    fi
fi

# ── GreenBoost kernel module ─────────────────────────────────────────────────
echo ""
echo "GreenBoost"
echo "────────────────────────────────────────────────────────────────"

check "greenboost kernel module loaded" lsmod_greenboost() { lsmod | grep -q greenboost; }; lsmod_greenboost

if [[ -f /etc/ld.so.preload ]]; then
    if grep -q libgreenboost_cuda /etc/ld.so.preload 2>/dev/null; then
        echo "  [PASS] ld.so.preload contains GreenBoost shim"
        (( PASS++ ))
    else
        echo "  [FAIL] ld.so.preload does not reference GreenBoost shim"
        (( FAIL++ ))
    fi
else
    echo "  [FAIL] /etc/ld.so.preload does not exist"
    (( FAIL++ ))
fi

SHIM_PATH=$(cat /etc/ld.so.preload 2>/dev/null | head -1)
if [[ -n "${SHIM_PATH}" && -f "${SHIM_PATH}" ]]; then
    echo "  [PASS] Shim file exists: ${SHIM_PATH}"
    (( PASS++ ))

    if nm -D "${SHIM_PATH}" 2>/dev/null | grep -q cuMemAlloc; then
        echo "  [PASS] Shim exports CUDA memory hooks"
        (( PASS++ ))
    else
        echo "  [FAIL] Shim does not export expected CUDA hooks"
        (( FAIL++ ))
    fi
fi

# ── System tuning ────────────────────────────────────────────────────────────
echo ""
echo "System Tuning"
echo "────────────────────────────────────────────────────────────────"

MEMLOCK=$(ulimit -l 2>/dev/null)
if [[ "${MEMLOCK}" == "unlimited" ]] || [[ "${MEMLOCK}" -gt 65536 ]] 2>/dev/null; then
    echo "  [PASS] memlock limit: ${MEMLOCK}"
    (( PASS++ ))
else
    echo "  [FAIL] memlock limit too low: ${MEMLOCK} (need unlimited)"
    (( FAIL++ ))
fi

check "/etc/security/limits.d/99-greenboost.conf exists" test -f /etc/security/limits.d/99-greenboost.conf

# ── vLLM ─────────────────────────────────────────────────────────────────────
echo ""
echo "vLLM Service"
echo "────────────────────────────────────────────────────────────────"

check "vllm.service exists" test -f /etc/systemd/system/vllm.service

if systemctl is-active --quiet vllm 2>/dev/null; then
    echo "  [PASS] vllm.service is active"
    (( PASS++ ))
else
    echo "  [FAIL] vllm.service is not active"
    (( FAIL++ ))
fi

if systemctl is-enabled --quiet vllm 2>/dev/null; then
    echo "  [PASS] vllm.service is enabled (starts on boot)"
    (( PASS++ ))
else
    warn "vllm.service is not enabled"
fi

# Check which GreenBoost path is in use
echo ""
echo "GreenBoost Overflow Path"
echo "────────────────────────────────────────────────────────────────"

VLLM_LOG=$(journalctl -u vllm --no-pager -n 200 2>/dev/null || true)

if echo "${VLLM_LOG}" | grep -q "DMA-BUF import"; then
    echo "  [PASS] Path A (DMA-BUF) active — optimal performance"
    (( PASS++ ))
elif echo "${VLLM_LOG}" | grep -q "HostReg alloc"; then
    echo "  [PASS] Path B (HostReg) active — good performance"
    (( PASS++ ))
elif echo "${VLLM_LOG}" | grep -q "UVM alloc"; then
    warn "Path C (UVM) active — this is 30-50x slower than Path A/B"
elif echo "${VLLM_LOG}" | grep -q "GREENBOOST"; then
    warn "GreenBoost is active but could not determine overflow path"
else
    warn "No GreenBoost log entries found — model may fit entirely in VRAM"
fi

# Check for known errors
if echo "${VLLM_LOG}" | grep -q "cuMemHostGetDevicePointer FAILED ret=201"; then
    echo "  [FAIL] cuMemHostGetDevicePointer v1 bug detected — apply the GreenBoost patch"
    (( FAIL++ ))
fi

if echo "${VLLM_LOG}" | grep -q "cudaErrorDeviceUninitialized"; then
    echo "  [FAIL] CUDA context init bug detected — apply the vLLM patch"
    (( FAIL++ ))
fi

# ── API endpoint ─────────────────────────────────────────────────────────────
echo ""
echo "API Endpoint"
echo "────────────────────────────────────────────────────────────────"

VLLM_PORT=$(grep -oP 'port \K[0-9]+' /etc/systemd/system/vllm.service 2>/dev/null || echo "8000")

if curl -sf "http://localhost:${VLLM_PORT}/health" &>/dev/null; then
    echo "  [PASS] vLLM /health endpoint responding on port ${VLLM_PORT}"
    (( PASS++ ))
elif curl -sf "http://localhost:${VLLM_PORT}/v1/models" &>/dev/null; then
    echo "  [PASS] vLLM /v1/models endpoint responding on port ${VLLM_PORT}"
    (( PASS++ ))
else
    warn "vLLM API not responding yet (model may still be loading)"
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Results:  ${PASS} passed,  ${FAIL} failed,  ${WARN} warnings"

if [[ $FAIL -gt 0 ]]; then
    echo "Some checks failed.  See README.md troubleshooting section."
    exit 1
fi
echo "All checks passed."
