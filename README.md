# vllm-greenboost

Run LLM models **larger than your GPU VRAM** by transparently overflowing GPU memory into system RAM — at PCIe bandwidth, not page-fault speed.

This repository documents a working setup of [vLLM](https://github.com/vllm-project/vllm) with [NVIDIA GreenBoost](https://gitlab.com/IsolatedOctopi/nvidia_greenboost) on Ubuntu, including **two upstream bug fixes** we discovered during the process and a fully automated setup script.

**Tested configuration**: Qwen2.5-14B FP16 (28 GB model) running on a Quadro RTX 5000 (16 GB VRAM) — 14 GB of model weights overflow to host RAM via direct PCIe DMA.

## Table of Contents

- [Background](#background)
- [How GreenBoost Works](#how-greenboost-works)
- [Bugs We Found and Fixed](#bugs-we-found-and-fixed)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Manual Setup](#manual-setup)
- [Configuration Reference](#configuration-reference)
- [Verification](#verification)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Upstream Status](#upstream-status)

## Background

Consumer and workstation GPUs often lack the VRAM needed to run modern LLMs at full precision. A 14B-parameter model in FP16 requires ~28 GB — well beyond most GPUs under $2,000.

The typical workaround is quantization (GGUF, AWQ, GPTQ), which reduces quality. Another option is CPU offloading, which uses unified virtual memory (UVM) page faults — functional but extremely slow (~30-50x overhead).

GreenBoost takes a different approach: it intercepts CUDA memory allocations via `LD_PRELOAD` and, when VRAM is exhausted, allocates host RAM and maps it into the GPU's address space using `cuMemHostRegister(CU_MEMHOSTREGISTER_DEVICEMAP)`. The GPU accesses this memory directly over the PCIe bus via DMA, without page faults. This gives you the full PCIe bandwidth (~16 GB/s on Gen3 x16, ~32 GB/s on Gen4, ~64 GB/s on Gen5) for the overflow portion.

The result: you can run models larger than your VRAM at reduced but usable speed, with no quantization and no code changes to the serving framework.

## How GreenBoost Works

GreenBoost has two components:

1. **Kernel module** (`greenboost.ko`) — manages a pool of host memory pages, pins them to prevent swap-out, and exports them to userspace via DMA-BUF file descriptors.
2. **CUDA shim** (`libgreenboost_cuda.so`) — loaded via `LD_PRELOAD`, intercepts `cuMemAlloc` and related calls. When VRAM is exhausted, it allocates from the kernel module's pool (or via `mmap`) and registers the host memory with CUDA.

The shim offers three overflow paths, tried in order:

| Path | Mechanism | Bandwidth | Requirements |
|------|-----------|-----------|--------------|
| **A — DMA-BUF** | Kernel-pinned pages + `cuMemHostRegister(DEVICEMAP)` | Full PCIe DMA | `greenboost.ko` loaded |
| **B — HostReg** | `mmap` + `cuMemHostRegister(DEVICEMAP)` | Full PCIe DMA | No kernel module needed |
| **C — UVM** | `cuMemAllocManaged` (unified virtual memory) | 30-50x slower | Last resort fallback |

Paths A and B deliver the same PCIe bandwidth. Path A additionally kernel-pins pages, which prevents the OS from swapping them out under memory pressure — important for production stability. Path C is a fallback that should be avoided.

## Bugs We Found and Fixed

Getting GreenBoost to work with vLLM required fixing two upstream bugs. Without these fixes, the system silently falls back to UVM (Path C) or crashes.

### 1. GreenBoost: `cuMemHostGetDevicePointer` v1 vs v2

**Symptom**: GreenBoost log shows `cuMemHostGetDevicePointer FAILED ret=201`. Paths A and B both fail silently, falling back to Path C (UVM).

**Root cause**: The CUDA driver API has versioned entry points. `cuMemHostGetDevicePointer` (v1) was introduced with explicit CUDA contexts (`cuCtxCreate`). When applications use *primary contexts* instead (`cuDevicePrimaryCtxRetain`) — as PyTorch, vLLM, and virtually all modern ML frameworks do — the v1 entry point returns `CUDA_ERROR_INVALID_CONTEXT (201)`.

The GreenBoost shim resolves the function pointer at init time via:

```c
real_cuMemHostGetDevicePointer = (pfn_cuMemHostGetDevicePointer) dlsym(libcuda, "cuMemHostGetDevicePointer");
```

This loads the v1 symbol. The fix is to load v2 instead:

```c
real_cuMemHostGetDevicePointer = (pfn_cuMemHostGetDevicePointer) dlsym(libcuda, "cuMemHostGetDevicePointer_v2");
```

The `_v2` variant works with both explicit and primary contexts.

**Patch**: [`patches/greenboost-v2-device-pointer.patch`](patches/greenboost-v2-device-pointer.patch)

### 2. vLLM: CUDA context uninitialized in EngineCore subprocess

**Symptom**: `cudaErrorDeviceUninitialized` crash when vLLM starts, specifically in the `EngineCore` worker subprocess.

**Root cause**: vLLM's `EngineCore` runs in a `multiprocessing.spawn`-ed subprocess. When the worker calls `torch.cuda.mem_get_info()` (via `MemorySnapshot`) to measure available VRAM, the CUDA context in the subprocess has not been fully initialized yet. Under normal conditions, this works because PyTorch lazily initializes CUDA. But with `LD_PRELOAD` shims like GreenBoost intercepting CUDA calls, the initialization sequence is disrupted.

**Fix**: Force CUDA context initialization by performing a small allocation before the `MemorySnapshot` call:

```python
# In vllm/v1/worker/gpu_worker.py, before MemorySnapshot:
torch.zeros(1, device=self.device)
```

**Patch**: [`patches/vllm-cuda-context-init.patch`](patches/vllm-cuda-context-init.patch)

### Additional Discovery: NVIDIA Open Kernel Module Limitation

The NVIDIA open kernel module (`nvidia-*-open` packages) does not support `cuMemHostRegister` on Turing-architecture GPUs (compute capability `sm_75`, e.g., RTX 2070/2080, Quadro RTX 4000/5000/6000/8000, T4). Attempting it returns `CUDA_ERROR_NOT_SUPPORTED (801)`.

You **must** use the proprietary (closed-source) NVIDIA driver for GreenBoost Path A/B to work on Turing GPUs. This limitation does not apply to Ampere (`sm_80`) and newer architectures where the open module fully supports `cuMemHostRegister`.

## Requirements

- **OS**: Ubuntu 22.04, 24.04, or 25.10 (tested on 25.10)
- **GPU**: NVIDIA GPU with CUDA support (tested: Quadro RTX 5000 / Turing `sm_75`)
- **Driver**: NVIDIA **proprietary** driver 535+ (tested: 590.48.01). The open kernel module will **not** work on Turing.
- **RAM**: System RAM must be larger than the model size. The overflow portion of the model lives in host memory.
- **PCIe**: Performance scales with PCIe generation. Gen3 x16 is the minimum for usable throughput.
- **Build tools**: `build-essential`, `git`, kernel headers (installed automatically by the setup script)

## Quick Start

```bash
git clone https://github.com/ktmb1/vllm-greenboost.git
cd vllm-greenboost

cp .env.example .env
# Edit .env — at minimum, review MODEL_ID and GPU_MEMORY_UTILIZATION

sudo ./setup.sh
```

The script is idempotent — safe to re-run after changing `.env`. If a new NVIDIA driver is installed, it will prompt you to reboot and re-run.

After setup:

```bash
# Verify the installation
./scripts/verify.sh

# Run a benchmark
./scripts/benchmark.sh
```

## Manual Setup

If you prefer to understand and run each step yourself, here is the full procedure.

### Step 1: Install the NVIDIA Proprietary Driver

Remove any open kernel module packages and install the proprietary driver:

```bash
DRIVER_VERSION=590
DRIVER_FLAVOR=server  # or "desktop" for X11 systems
KERNEL_VER=$(uname -r)

# Remove open kernel module if present
sudo apt-get remove -y --purge \
    nvidia-headless-no-dkms-${DRIVER_VERSION}-${DRIVER_FLAVOR}-open \
    nvidia-kernel-source-${DRIVER_VERSION}-${DRIVER_FLAVOR}-open

# Install proprietary driver
sudo apt-get update
sudo apt-get install -y \
    nvidia-headless-no-dkms-${DRIVER_VERSION}-${DRIVER_FLAVOR} \
    linux-modules-nvidia-${DRIVER_VERSION}-${DRIVER_FLAVOR}-${KERNEL_VER} \
    libnvidia-compute-${DRIVER_VERSION}-${DRIVER_FLAVOR} \
    nvidia-compute-utils-${DRIVER_VERSION}-${DRIVER_FLAVOR} \
    nvidia-utils-${DRIVER_VERSION}-${DRIVER_FLAVOR}

sudo apt-get install -y nvidia-cuda-toolkit

# REBOOT after driver install
sudo reboot
```

Verify the driver is proprietary (not GPL/open):

```bash
modinfo /lib/modules/$(uname -r)/kernel/nvidia-*/nvidia.ko | grep '^license'
# Should show "NVIDIA" or "Proprietary", NOT "GPL"
```

### Step 2: Build and Patch GreenBoost

```bash
sudo apt-get install -y build-essential git linux-headers-$(uname -r)

sudo git clone https://gitlab.com/IsolatedOctopi/nvidia_greenboost.git /opt/greenboost
cd /opt/greenboost

# Apply the cuMemHostGetDevicePointer_v2 fix
sed -i 's/dlsym(libcuda, "cuMemHostGetDevicePointer")/dlsym(libcuda, "cuMemHostGetDevicePointer_v2")/' \
    greenboost_cuda_shim.c

# Build
make module
make shim
sudo make install

# Install the shim library
sudo cp libgreenboost_cuda.so /usr/local/lib/libgreenboost_cuda.so
sudo chmod 755 /usr/local/lib/libgreenboost_cuda.so
sudo ldconfig
```

### Step 3: Configure the System

```bash
# Detect GPU VRAM and system RAM
GPU_VRAM_GB=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) / 1024 ))
TOTAL_RAM_GB=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
VIRTUAL_VRAM_GB=$(( TOTAL_RAM_GB * 65 / 100 ))

# GreenBoost kernel module configuration
echo "options greenboost physical_vram_gb=${GPU_VRAM_GB} virtual_vram_gb=${VIRTUAL_VRAM_GB} nvme_swap_gb=0" \
    | sudo tee /etc/modprobe.d/greenboost.conf

echo "softdep greenboost pre: nvidia nvidia_uvm" \
    | sudo tee /etc/modprobe.d/greenboost-softdep.conf

echo "greenboost" | sudo tee /etc/modules-load.d/greenboost.conf

# Set unlimited memlock (required for cuMemHostRegister)
cat <<'EOF' | sudo tee /etc/security/limits.d/99-greenboost.conf
* soft memlock unlimited
* hard memlock unlimited
EOF

# Enable the GreenBoost shim globally
echo "/usr/local/lib/libgreenboost_cuda.so" | sudo tee /etc/ld.so.preload

# Load the kernel module
sudo modprobe greenboost
```

### Step 4: Install and Patch vLLM

```bash
sudo apt-get install -y python3 python3-venv python3-pip

# Create service user
sudo useradd --system --no-create-home --shell /usr/sbin/nologin --groups video vllm

# Create venv and install vLLM
sudo python3 -m venv /opt/vllm-env
sudo /opt/vllm-env/bin/pip install vllm==0.18.0

# Apply the CUDA context initialization patch
GPU_WORKER=$(find /opt/vllm-env -path '*/vllm/v1/worker/gpu_worker.py' -type f | head -1)
sed -i '/^\s*# take current memory snapshot/i\
            # Ensure CUDA context is fully initialized (needed for LD_PRELOAD shims)\
            torch.zeros(1, device=self.device)\
' "${GPU_WORKER}"

sudo chown -R vllm:video /opt/vllm-env
```

### Step 5: Download a Model

```bash
sudo mkdir -p /opt/models/qwen2.5-14b-instruct
sudo chown vllm:video /opt/models/qwen2.5-14b-instruct

sudo -u vllm /opt/vllm-env/bin/python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-14B-Instruct', local_dir='/opt/models/qwen2.5-14b-instruct')
"
```

### Step 6: Create and Start the systemd Service

Create `/etc/systemd/system/vllm.service`:

```ini
[Unit]
Description=vLLM Model Server (GreenBoost VRAM oversubscription)
After=network.target nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
Type=exec
User=vllm
Group=video

Environment=LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so
Environment=GREENBOOST_ACTIVE=1
Environment=GREENBOOST_DEBUG=0
Environment=GREENBOOST_VRAM_HEADROOM_MB=2048
Environment=GREENBOOST_USE_DMA_BUF=1
Environment=GREENBOOST_NO_HOSTREG=0
Environment=CUDA_VISIBLE_DEVICES=0

ExecStart=/opt/vllm-env/bin/vllm serve /opt/models/qwen2.5-14b-instruct \
    --dtype half \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.5 \
    --host 0.0.0.0 \
    --port 8000 \
    --enforce-eager

Restart=on-failure
RestartSec=10
LimitMEMLOCK=infinity
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now vllm
```

## Configuration Reference

All configuration is in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `NVIDIA_DRIVER_VERSION` | `590` | NVIDIA driver branch |
| `NVIDIA_DRIVER_FLAVOR` | `server` | `server` (headless) or `desktop` (X11) |
| `GREENBOOST_REPO` | GitLab URL | GreenBoost git repository |
| `GREENBOOST_BRANCH` | `main` | Git branch to clone |
| `GREENBOOST_INSTALL_DIR` | `/opt/greenboost` | Clone destination |
| `GREENBOOST_SHIM_PATH` | `/usr/local/lib/libgreenboost_cuda.so` | Installed shim location |
| `GREENBOOST_VRAM_HEADROOM_MB` | `2048` | VRAM reserved for CUDA overhead |
| `GREENBOOST_USE_DMA_BUF` | `1` | `1` = Path A (kernel pinning), `0` = Path B |
| `VLLM_VERSION` | `0.18.0` | vLLM pip version |
| `VLLM_VENV` | `/opt/vllm-env` | Python venv location |
| `MODEL_ID` | `Qwen/Qwen2.5-14B-Instruct` | HuggingFace model ID |
| `MODEL_PATH` | `/opt/models/qwen2.5-14b-instruct` | Local model storage |
| `VLLM_DTYPE` | `half` | Precision: `half`, `bfloat16`, or `auto` |
| `VLLM_MAX_MODEL_LEN` | `2048` | Maximum sequence length |
| `GPU_MEMORY_UTILIZATION` | `0.5` | Fraction of reported VRAM to use |
| `VLLM_HOST` | `0.0.0.0` | Listen address |
| `VLLM_PORT` | `8000` | Listen port |
| `VLLM_USER` | `vllm` | System user for the service |
| `VLLM_GROUP` | `video` | System group (needs GPU access) |

### Understanding `GPU_MEMORY_UTILIZATION`

GreenBoost inflates the VRAM reported to CUDA. For example, a 16 GB GPU with 60 GB of host RAM will report ~66 GB of total VRAM. vLLM's `--gpu-memory-utilization` controls what fraction of this *reported* VRAM it tries to use.

Setting this to `0.95` (the vLLM default) would make it try to allocate ~63 GB — more than physical RAM — triggering the OOM killer.

**Rule of thumb**: `(model_size_gb + desired_kv_cache_gb) / reported_vram_gb`

For a 28 GB model on a 16 GB GPU with 60 GB RAM: `0.5` works well, giving ~33 GB for weights + KV cache.

## Verification

After setup, run the verification script:

```bash
./scripts/verify.sh
```

It checks:
- NVIDIA driver is proprietary (not open kernel module)
- GreenBoost kernel module is loaded
- GreenBoost shim is in `ld.so.preload`
- `memlock` limit is unlimited
- vLLM systemd service is active
- GreenBoost is using Path A or B (not C)
- No known error patterns in logs
- vLLM API endpoint is responding

You can also verify manually:

```bash
# Check GreenBoost overflow path in vLLM logs
journalctl -u vllm | grep -E "DMA-BUF import|HostReg alloc|UVM alloc"
# You want to see "DMA-BUF import" (Path A) or "HostReg alloc" (Path B)
# NOT "UVM alloc" (Path C — 30-50x slower)

# Test inference
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/opt/models/qwen2.5-14b-instruct", "prompt": "Hello!", "max_tokens": 50}'
```

## Performance

Performance is bounded by PCIe bandwidth for the overflow portion of the model. Each token generation reads the full model weights — those in VRAM at full GPU bandwidth, those overflowed to host RAM at PCIe bandwidth.

### Measured Results

| Model | GPU | VRAM | Overflow | PCIe | Throughput |
|-------|-----|------|----------|------|------------|
| Qwen2.5-14B FP16 (28 GB) | Quadro RTX 5000 | 16 GB | ~14 GB → host RAM | Gen3 x16 | ~0.73 tok/s |

### Expected Scaling

| PCIe Generation | Bandwidth | Approximate Throughput Multiplier |
|-----------------|-----------|-----------------------------------|
| Gen3 x16 | ~16 GB/s | 1x (baseline) |
| Gen4 x16 | ~32 GB/s | ~2x |
| Gen5 x16 | ~64 GB/s | ~4x |

Higher-end GPUs with more VRAM will overflow less to host RAM, giving a disproportionate speedup. For example, a 24 GB GPU running the same 28 GB model only overflows 4 GB instead of 14 GB.

### When to Use This vs. Quantization

| Approach | Quality | Speed | VRAM Needed |
|----------|---------|-------|-------------|
| Full-precision + GreenBoost | Original | Slower (PCIe-bound) | GPU VRAM + host RAM |
| 4-bit quantization (GGUF/AWQ) | Reduced | Fast (fits in VRAM) | ~25% of original |
| CPU offload (UVM/Path C) | Original | Very slow | GPU VRAM + host RAM |

GreenBoost is best when you need full-precision inference and have sufficient host RAM, but your GPU lacks the VRAM to fit the model. It trades throughput for quality.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `cuMemHostGetDevicePointer FAILED ret=201` | Shim uses v1 entry point | Apply the [GreenBoost patch](patches/greenboost-v2-device-pointer.patch) |
| `Path B (HostReg) failed` + UVM fallback | Open kernel module **or** memlock limit | Use proprietary driver; check `ulimit -l` is unlimited |
| `CUDA_ERROR_NOT_SUPPORTED (801)` on `cuMemHostRegister` | Open kernel module on Turing GPU | Switch to proprietary driver |
| `cudaErrorDeviceUninitialized` in EngineCore | CUDA context not initialized in subprocess | Apply the [vLLM patch](patches/vllm-cuda-context-init.patch) |
| OOM killer kills vLLM | `GPU_MEMORY_UTILIZATION` too high | Lower to 0.5 or calculate per formula above |
| Model download hangs | IPv6 CDN connectivity issues | Set `HF_HUB_DISABLE_XET=1`; the setup script forces IPv4 |
| `sudo` or `cp` segfaults | Broken shim in `/etc/ld.so.preload` | Clear preload (`echo "" > /etc/ld.so.preload`), fix shim, restore |
| GreenBoost log says "UVM alloc" | Path A and B failed | Check all of the above; enable `GREENBOOST_DEBUG=1` |

To enable debug logging:

```bash
# Temporarily, in the service environment:
sudo systemctl set-environment GREENBOOST_DEBUG=1
sudo systemctl restart vllm
journalctl -u vllm -f
```

## Upstream Status

Both bugs have been reported upstream with patches:

| Project | Issue | Fix | Status |
|---------|-------|-----|--------|
| GreenBoost | `cuMemHostGetDevicePointer` v1 incompatible with primary context | [MR !5](https://gitlab.com/IsolatedOctopi/nvidia_greenboost/-/merge_requests/5) | Pending review |
| vLLM | `MemorySnapshot` before CUDA context init in subprocess | [PR #38417](https://github.com/vllm-project/vllm/pull/38417) | Pending review |

Once these are merged upstream, the patches in this repository will no longer be needed and the setup script will skip applying them.

## License

[MIT](LICENSE)

## Acknowledgments

- [NVIDIA GreenBoost](https://gitlab.com/IsolatedOctopi/nvidia_greenboost) by IsolatedOctopi — the core VRAM oversubscription technology
- [vLLM](https://github.com/vllm-project/vllm) — the high-throughput LLM serving engine
