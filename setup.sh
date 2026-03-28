#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# vllm-greenboost setup script
#
# Turns a fresh Ubuntu server with an NVIDIA GPU into an LLM inference server
# that can run models LARGER than physical VRAM using GreenBoost + vLLM.
#
# Usage:
#   cp .env.example .env   # edit to taste
#   sudo ./setup.sh
#
# The script is idempotent — safe to re-run after changing .env.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

unset LD_PRELOAD 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Load configuration ────────────────────────────────────────────────────────
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    # shellcheck source=.env.example
    source "${SCRIPT_DIR}/.env"
else
    echo "ERROR: No .env file found.  Copy .env.example to .env and configure it."
    echo "       cp .env.example .env"
    exit 1
fi

# ── Preflight checks ─────────────────────────────────────────────────────────
section() { echo -e "\n\033[1;36m══ $1 ══\033[0m"; }

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: This script must be run as root (sudo ./setup.sh)."
    exit 1
fi

section "Preflight checks"

if ! grep -qi ubuntu /etc/os-release 2>/dev/null; then
    echo "WARNING: This script is tested on Ubuntu 22.04/24.04/25.10."
    echo "         Proceeding anyway — your mileage may vary."
fi

if ! lspci | grep -qi nvidia; then
    echo "ERROR: No NVIDIA GPU detected (lspci).  Is the GPU passed through / installed?"
    exit 1
fi

TOTAL_RAM_GB=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
echo "Detected ${TOTAL_RAM_GB} GB system RAM."
echo "GPU(s): $(lspci | grep -i 'vga\|3d' | grep -i nvidia | head -1)"

# ── 1. NVIDIA proprietary driver ─────────────────────────────────────────────
section "NVIDIA proprietary driver (${NVIDIA_DRIVER_VERSION}-${NVIDIA_DRIVER_FLAVOR})"

KERNEL_VER=$(uname -r)

NEED_DRIVER=0
if ! command -v nvidia-smi &>/dev/null; then
    NEED_DRIVER=1
elif nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | grep -q "^${NVIDIA_DRIVER_VERSION}"; then
    echo "Driver ${NVIDIA_DRIVER_VERSION} already installed."
else
    NEED_DRIVER=1
fi

if [[ $NEED_DRIVER -eq 1 ]]; then
    echo "Removing any open-kernel-module packages..."
    apt-get remove -y --purge \
        "nvidia-headless-no-dkms-${NVIDIA_DRIVER_VERSION}-${NVIDIA_DRIVER_FLAVOR}-open" \
        "nvidia-kernel-source-${NVIDIA_DRIVER_VERSION}-${NVIDIA_DRIVER_FLAVOR}-open" \
        2>/dev/null || true

    echo "Installing proprietary driver packages..."
    apt-get update -qq
    apt-get install -y \
        "nvidia-headless-no-dkms-${NVIDIA_DRIVER_VERSION}-${NVIDIA_DRIVER_FLAVOR}" \
        "linux-modules-nvidia-${NVIDIA_DRIVER_VERSION}-${NVIDIA_DRIVER_FLAVOR}-${KERNEL_VER}" \
        "libnvidia-compute-${NVIDIA_DRIVER_VERSION}-${NVIDIA_DRIVER_FLAVOR}" \
        "nvidia-compute-utils-${NVIDIA_DRIVER_VERSION}-${NVIDIA_DRIVER_FLAVOR}" \
        "nvidia-utils-${NVIDIA_DRIVER_VERSION}-${NVIDIA_DRIVER_FLAVOR}"

    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  NVIDIA driver installed.  A REBOOT is required before the     ║"
    echo "║  remaining steps can succeed.  Re-run this script after reboot.║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    DRIVER_JUST_INSTALLED=1
fi

# Build-time only: provides nvcc and headers for compiling the GreenBoost
# kernel module + shim.  Runtime CUDA libs come from PyTorch's pip packages;
# LD_LIBRARY_PATH in the systemd service ensures those take priority.
apt-get install -y nvidia-cuda-toolkit 2>/dev/null || true

if [[ "${DRIVER_JUST_INSTALLED:-0}" -eq 1 ]]; then
    echo ""
    echo "Reboot now?  (sudo reboot)"
    exit 0
fi

# Verify driver is proprietary (not open kernel module) — required on Turing.
DRIVER_LICENSE=$(modinfo nvidia 2>/dev/null | awk '/^license:/ {print $2}')
if [[ "${DRIVER_LICENSE}" == "GPL" ]]; then
    GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    if [[ "${GPU_CC}" == "7.5" ]]; then
        echo "ERROR: The loaded nvidia.ko is GPL (open kernel module)."
        echo "       cuMemHostRegister is not supported on Turing (sm_75) with the open module."
        echo "       Remove open packages and install proprietary ones."
        exit 1
    else
        echo "WARNING: Open kernel module detected (license: GPL)."
        echo "         This is OK for Ampere+ (compute ${GPU_CC}), but if you hit"
        echo "         CUDA_ERROR_NOT_SUPPORTED (801), switch to the proprietary driver."
    fi
else
    echo "Driver verified: proprietary (license: ${DRIVER_LICENSE:-NVIDIA})."
fi

# ── 2. GreenBoost ────────────────────────────────────────────────────────────
section "GreenBoost (clone, patch, build)"

apt-get install -y build-essential git "linux-headers-${KERNEL_VER}" 2>/dev/null || true

if [[ -d "${GREENBOOST_INSTALL_DIR}/.git" ]]; then
    echo "Updating existing clone..."
    git -C "${GREENBOOST_INSTALL_DIR}" fetch origin
    git -C "${GREENBOOST_INSTALL_DIR}" reset --hard "origin/${GREENBOOST_BRANCH}"
else
    echo "Cloning GreenBoost..."
    git clone --branch "${GREENBOOST_BRANCH}" "${GREENBOOST_REPO}" "${GREENBOOST_INSTALL_DIR}"
fi

echo "Applying cuMemHostGetDevicePointer_v2 patch..."
SHIM_SRC="${GREENBOOST_INSTALL_DIR}/greenboost_cuda_shim.c"
if grep -q 'cuMemHostGetDevicePointer"' "${SHIM_SRC}" 2>/dev/null; then
    sed -i 's/dlsym(libcuda, "cuMemHostGetDevicePointer")/dlsym(libcuda, "cuMemHostGetDevicePointer_v2")/' "${SHIM_SRC}"
    echo "Patch applied."
elif grep -q 'cuMemHostGetDevicePointer_v2' "${SHIM_SRC}" 2>/dev/null; then
    echo "Patch already applied (upstream may have merged it)."
else
    echo "WARNING: Could not find the expected dlsym line.  Trying git apply..."
    cd "${GREENBOOST_INSTALL_DIR}"
    git apply "${SCRIPT_DIR}/patches/greenboost-v2-device-pointer.patch" || true
fi

# Clear ld.so.preload before building — on re-runs the old shim is loaded
# into every process (make, gcc, insmod …) and segfaults if it tries to
# intercept CUDA calls in a non-CUDA context.
LD_PRELOAD_BAK=""
if [[ -f /etc/ld.so.preload ]]; then
    LD_PRELOAD_BAK=$(cat /etc/ld.so.preload)
    echo "" > /etc/ld.so.preload
    echo "Temporarily cleared /etc/ld.so.preload for clean build."
fi

echo "Building kernel module..."
make -C "${GREENBOOST_INSTALL_DIR}" module

echo "Building CUDA shim..."
make -C "${GREENBOOST_INSTALL_DIR}" shim

echo "Installing kernel module..."
make -C "${GREENBOOST_INSTALL_DIR}" install

# Re-clear: GreenBoost's Makefile may restore /etc/ld.so.preload during install
echo "" > /etc/ld.so.preload 2>/dev/null || true

echo "Installing shim to ${GREENBOOST_SHIM_PATH}..."
cp "${GREENBOOST_INSTALL_DIR}/libgreenboost_cuda.so" "${GREENBOOST_SHIM_PATH}"
chmod 755 "${GREENBOOST_SHIM_PATH}"
ldconfig

# ── 3. System configuration ──────────────────────────────────────────────────
section "System configuration"

# Detect GPU VRAM
GPU_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GB_PHYSICAL_VRAM_GB=$(( GPU_VRAM_MB / 1024 ))
GB_VIRTUAL_VRAM_GB=$(( TOTAL_RAM_GB * 65 / 100 ))
export GB_PHYSICAL_VRAM_GB GB_VIRTUAL_VRAM_GB

echo "GPU VRAM: ${GB_PHYSICAL_VRAM_GB} GB physical, ${GB_VIRTUAL_VRAM_GB} GB virtual (65% of ${TOTAL_RAM_GB} GB RAM)"

# modprobe configuration
envsubst '${GB_PHYSICAL_VRAM_GB} ${GB_VIRTUAL_VRAM_GB}' \
    < "${SCRIPT_DIR}/config/greenboost-modprobe.conf" \
    > /etc/modprobe.d/greenboost.conf

cp "${SCRIPT_DIR}/config/greenboost-softdep.conf" /etc/modprobe.d/greenboost-softdep.conf
cp "${SCRIPT_DIR}/config/modules-load-greenboost.conf" /etc/modules-load.d/greenboost.conf

# memlock limits (required for cuMemHostRegister)
cp "${SCRIPT_DIR}/config/99-greenboost-memlock.conf" /etc/security/limits.d/99-greenboost.conf

# Load module now (before writing ld.so.preload — modprobe can segfault
# if the shim is globally preloaded into it).
rmmod greenboost 2>/dev/null || true
modprobe greenboost
echo "GreenBoost kernel module loaded."

# Global LD_PRELOAD — enable only after all system commands are done.
echo "${GREENBOOST_SHIM_PATH}" > /etc/ld.so.preload

# ── 4. vLLM ──────────────────────────────────────────────────────────────────
section "vLLM ${VLLM_VERSION}"

apt-get install -y python3 python3-venv python3-pip 2>/dev/null || true

# Create service user
if ! id "${VLLM_USER}" &>/dev/null; then
    useradd --system --no-create-home --shell /usr/sbin/nologin \
        --groups "${VLLM_GROUP}" "${VLLM_USER}"
    echo "Created system user ${VLLM_USER}."
fi

# Create virtual environment
if [[ ! -f "${VLLM_VENV}/bin/activate" ]]; then
    python3 -m venv "${VLLM_VENV}"
    echo "Created venv at ${VLLM_VENV}."
fi

echo "Installing vLLM ${VLLM_VERSION} (this may take a few minutes)..."
"${VLLM_VENV}/bin/pip" install --quiet "vllm==${VLLM_VERSION}"

# Apply CUDA context initialization patch
echo "Patching vLLM gpu_worker.py..."
GPU_WORKER=$(find "${VLLM_VENV}" -path '*/vllm/v1/worker/gpu_worker.py' -type f | head -1)

if [[ -z "${GPU_WORKER}" ]]; then
    echo "WARNING: Could not locate gpu_worker.py — skipping vLLM patch."
else
    if grep -q 'torch.zeros(1, device=self.device)' "${GPU_WORKER}"; then
        echo "vLLM patch already applied."
    else
        # Insert torch.zeros before the MemorySnapshot line
        sed -i '/^\s*# take current memory snapshot/i\
            # Ensure CUDA context is fully initialized (needed for LD_PRELOAD shims)\
            torch.zeros(1, device=self.device)\
' "${GPU_WORKER}"
        echo "vLLM patch applied to ${GPU_WORKER}."
    fi
fi

# The nvidia-cuda-runtime-cu12 pip package ships libcudart.so.12 but not the
# unversioned libcudart.so symlink.  Some libraries in the dependency chain
# load "libcudart.so" (unversioned); without this symlink the dynamic linker
# falls through to the system's older libcudart from nvidia-cuda-toolkit,
# poisoning symbol resolution for the rest of the process.
CUDART_DIR=$("${VLLM_VENV}/bin/python3" -c "
import importlib.util, pathlib
base = pathlib.Path(importlib.util.find_spec('nvidia').submodule_search_locations[0])
print(base / 'cuda_runtime' / 'lib')
")
if [[ -f "${CUDART_DIR}/libcudart.so.12" && ! -e "${CUDART_DIR}/libcudart.so" ]]; then
    ln -sf libcudart.so.12 "${CUDART_DIR}/libcudart.so"
    echo "Created libcudart.so symlink in ${CUDART_DIR}."
fi

# Build LD_LIBRARY_PATH so PyTorch's bundled CUDA libs take priority over
# the system nvidia-cuda-toolkit (which may be an older CUDA version).
TORCH_LIB=$("${VLLM_VENV}/bin/python3" -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')")
NVIDIA_LIBS=$("${VLLM_VENV}/bin/python3" -c "
import importlib.util, pathlib
base = pathlib.Path(importlib.util.find_spec('nvidia').submodule_search_locations[0])
print(':'.join(str(p) for p in sorted(base.glob('*/lib'))))
" 2>/dev/null || true)
VLLM_LD_LIBRARY_PATH="${TORCH_LIB}${NVIDIA_LIBS:+:${NVIDIA_LIBS}}"
export VLLM_LD_LIBRARY_PATH
echo "PyTorch CUDA library path: ${VLLM_LD_LIBRARY_PATH}"

chown -R "${VLLM_USER}:${VLLM_GROUP}" "${VLLM_VENV}"

# ── 5. Model download ────────────────────────────────────────────────────────
section "Model: ${MODEL_ID}"

mkdir -p "${MODEL_PATH}"
chown "${VLLM_USER}:${VLLM_GROUP}" "${MODEL_PATH}"

if [[ -f "${MODEL_PATH}/config.json" ]]; then
    echo "Model already downloaded at ${MODEL_PATH}."
else
    echo "Downloading model from HuggingFace (this may take a while)..."
    # Force IPv4 to avoid IPv6 CDN connectivity issues on some networks
    sudo -u "${VLLM_USER}" "${VLLM_VENV}/bin/python3" -c "
import socket
_orig = socket.getaddrinfo
def _ipv4_only(*args, **kwargs):
    return [r for r in _orig(*args, **kwargs) if r[0] == socket.AF_INET]
socket.getaddrinfo = _ipv4_only
import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import snapshot_download
snapshot_download('${MODEL_ID}', local_dir='${MODEL_PATH}')
"
    echo "Model downloaded."
fi

# ── 6. systemd service ───────────────────────────────────────────────────────
section "systemd service"

# Export all variables needed by the service template
export GREENBOOST_SHIM_PATH GREENBOOST_VRAM_HEADROOM_MB GREENBOOST_USE_DMA_BUF
export VLLM_VENV VLLM_LD_LIBRARY_PATH
export MODEL_PATH VLLM_DTYPE VLLM_MAX_MODEL_LEN GPU_MEMORY_UTILIZATION
export VLLM_HOST VLLM_PORT VLLM_USER VLLM_GROUP

envsubst < "${SCRIPT_DIR}/config/vllm.service" > /etc/systemd/system/vllm.service

systemctl daemon-reload
systemctl enable vllm
systemctl restart vllm

echo "vLLM service started.  Waiting for it to become ready..."
sleep 5

if systemctl is-active --quiet vllm; then
    echo "vLLM service is running."
else
    echo "WARNING: vLLM service is not running.  Check: journalctl -u vllm -n 50"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
section "Setup complete"

echo ""
echo "Verify the setup:"
echo "  ./scripts/verify.sh"
echo ""
echo "Test inference:"
echo "  curl -s http://localhost:${VLLM_PORT}/v1/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"${MODEL_PATH}\", \"prompt\": \"Hello!\", \"max_tokens\": 50}'"
echo ""
echo "Benchmark:"
echo "  ./scripts/benchmark.sh"
