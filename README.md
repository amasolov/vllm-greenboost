# vllm-greenboost

vLLM + [NVIDIA GreenBoost](https://gitlab.com/IsolatedOctopi/nvidia_greenboost) VRAM oversubscription: run models larger than GPU VRAM by overflowing allocations to host RAM via PCIe DMA (`cuMemHostRegister(DEVICEMAP)`), not UVM page faults.

## Tested Setup

| Component | Value |
|-----------|-------|
| OS | Ubuntu 25.10 |
| GPU | Quadro RTX 5000 (Turing, `sm_75`, 16 GB VRAM) |
| Driver | NVIDIA 590.48.01 **proprietary** (not open kernel module) |
| CUDA toolkit | 12.4 |
| RAM | 60 GB DDR4 |
| PCIe | Gen3 x16 |
| vLLM | 0.18.0 |
| GreenBoost | v2.5 (main branch + patch below) |
| Model | Qwen2.5-14B-Instruct FP16 (~28 GB) |

## Test Results

The model requires ~28 GB but the GPU only has 16 GB VRAM. GreenBoost overflows ~14 GB to host RAM via DMA-BUF (Path A).

| Metric | Value |
|--------|-------|
| Model size | ~28 GB (FP16) |
| VRAM used | 16 GB (full GPU) |
| Host RAM overflow | ~14 GB (via `cuMemHostRegister` + DMA-BUF) |
| GreenBoost path | **A (DMA-BUF)** — kernel-pinned, PCIe DMA |
| Generation throughput | **~0.73 tok/s** |
| Bottleneck | PCIe 3.0 x16 bandwidth (~16 GB/s) for the overflow portion |

Without GreenBoost, this model cannot load at all on this GPU (immediate OOM). With the unpatched shim, it silently falls back to UVM (Path C) at ~30-50x worse throughput.

Expected throughput scaling with faster PCIe:

| PCIe | Bandwidth | Est. Throughput |
|------|-----------|-----------------|
| Gen3 x16 | ~16 GB/s | ~0.7 tok/s |
| Gen4 x16 | ~32 GB/s | ~1.4 tok/s |
| Gen5 x16 | ~64 GB/s | ~2.8 tok/s |

## GPU Compatibility

Both patches are **architecture-independent** — any GPU using PyTorch/vLLM hits the same bugs. The only architecture-specific difference is the driver requirement:

| | Turing (`sm_75`) | Ampere (`sm_80`) | Ada/Hopper+ |
|---|---|---|---|
| **GPUs** | RTX 2070-2080, Quadro RTX 4000-8000, T4 | RTX 3060-3090, A6000, A100 | RTX 4060-4090, L40, H100 |
| **Open driver works?** | No — must use proprietary | Yes | Yes |
| **PCIe gen** | 3.0 | 4.0 | 4.0 / 5.0 |
| **Both patches needed?** | Yes | Yes | Yes |

Estimated throughput for a **28 GB FP16 model** (e.g., Qwen2.5-14B) with sufficient host RAM:

| GPU | VRAM | Overflow | PCIe | Est. Throughput |
|-----|------|----------|------|-----------------|
| Quadro RTX 5000 | 16 GB | 14 GB | Gen3 x16 | ~0.7 tok/s (measured) |
| RTX 3090 | 24 GB | 4 GB | Gen4 x16 | ~3-5 tok/s |
| A6000 | 48 GB | 0 GB | Gen4 x16 | Full speed (fits in VRAM) |
| RTX 4090 | 24 GB | 4 GB | Gen4 x16 | ~3-5 tok/s |

GreenBoost is most valuable for models **significantly** larger than VRAM — e.g., 70B FP16 (~140 GB) on an A6000 (48 GB), overflowing ~92 GB to host RAM.

## Two Upstream Bugs Fixed

### 1. GreenBoost: `cuMemHostGetDevicePointer` v1 vs v2

The shim resolves `cuMemHostGetDevicePointer` (v1) via `dlsym`, which returns `CUDA_ERROR_INVALID_CONTEXT (201)` under primary contexts (`cuDevicePrimaryCtxRetain`) used by PyTorch/vLLM. This silently kills Path A and B, forcing UVM fallback.

Fix: `dlsym(libcuda, "cuMemHostGetDevicePointer")` -> `dlsym(libcuda, "cuMemHostGetDevicePointer_v2")`

- Patch: [`patches/greenboost-v2-device-pointer.patch`](patches/greenboost-v2-device-pointer.patch)
- Upstream: [GreenBoost MR !5](https://gitlab.com/IsolatedOctopi/nvidia_greenboost/-/merge_requests/5)

### 2. vLLM: CUDA context uninitialized in EngineCore subprocess

vLLM's `EngineCore` subprocess calls `torch.cuda.mem_get_info()` via `MemorySnapshot` before the CUDA context is initialized in the child process. With `LD_PRELOAD` shims active, this crashes with `cudaErrorDeviceUninitialized`.

Fix: add `torch.zeros(1, device=self.device)` before `MemorySnapshot` in `vllm/v1/worker/gpu_worker.py`.

- Patch: [`patches/vllm-cuda-context-init.patch`](patches/vllm-cuda-context-init.patch)
- Upstream: [vLLM PR #38417](https://github.com/vllm-project/vllm/pull/38417)

### Additional: NVIDIA Open Kernel Module on Turing

The open kernel module (`nvidia-*-open`) does not support `cuMemHostRegister` on Turing (`sm_75`) — returns `CUDA_ERROR_NOT_SUPPORTED (801)`. Proprietary driver required. Does not apply to Ampere+.

## Quick Start

```bash
git clone https://github.com/amasolov/vllm-greenboost.git
cd vllm-greenboost
cp .env.example .env   # edit MODEL_ID, GPU_MEMORY_UTILIZATION, etc.
sudo ./setup.sh
```

The script installs the proprietary NVIDIA driver, builds/patches GreenBoost, installs/patches vLLM, downloads the model, and starts a systemd service. Idempotent; reboot required after first driver install.

Post-install: `./scripts/verify.sh` checks driver type, module state, overflow path, and API health.

## Key Config: `GPU_MEMORY_UTILIZATION`

GreenBoost inflates reported VRAM (e.g., 16 GB GPU + 60 GB RAM reports ~66 GB). vLLM's `--gpu-memory-utilization` is a fraction of *reported* VRAM. Setting 0.95 (default) would try to allocate ~63 GB, triggering OOM.

Formula: `(model_size_gb + kv_cache_gb) / reported_vram_gb`. For 28 GB model on 16 GB GPU + 60 GB RAM: **0.5** works.

## Prerequisites

| Requirement | Tested Version | Notes |
|-------------|---------------|-------|
| OS | **Ubuntu 25.10** | 22.04 and 24.04 should work; not tested |
| NVIDIA driver | **590** (proprietary) | Must be closed-source, not `nvidia-*-open`. Open module lacks `cuMemHostRegister` on Turing (`sm_75`). Ampere+ may work with open. |
| CUDA toolkit | **12.4** | Installed via `nvidia-cuda-toolkit` |
| Python | **3.13** | 3.10+ should work |
| vLLM | **0.18.0** | Patch targets `vllm/v1/worker/gpu_worker.py` |
| GreenBoost | **main** branch | With `cuMemHostGetDevicePointer_v2` patch applied |
| RAM | **> model size** | Overflow portion lives in host memory. 60 GB tested for 28 GB model. |
| `ulimit -l` | **unlimited** | Required for `cuMemHostRegister`. Setup script configures this. |
| Build tools | `build-essential`, `git`, `linux-headers-$(uname -r)` | For GreenBoost kernel module compilation |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `cuMemHostGetDevicePointer FAILED ret=201` | Apply GreenBoost patch (v1 -> v2) |
| `CUDA_ERROR_NOT_SUPPORTED (801)` | Switch from open to proprietary driver |
| `cudaErrorDeviceUninitialized` | Apply vLLM patch |
| OOM killer | Lower `GPU_MEMORY_UTILIZATION` |
| Logs show `UVM alloc` instead of `DMA-BUF import` | All of the above; check `ulimit -l`; enable `GREENBOOST_DEBUG=1` |

## Repo Structure

```
patches/           # Both upstream fixes as unified diffs
config/            # systemd unit, modprobe, memlock templates
scripts/           # verify.sh, benchmark.sh
setup.sh           # Automated setup (sources .env)
.env.example       # All configurable parameters
```

## License

[MIT](LICENSE)
