# vllm-greenboost

vLLM + [NVIDIA GreenBoost](https://gitlab.com/IsolatedOctopi/nvidia_greenboost) VRAM oversubscription: run models larger than GPU VRAM by overflowing allocations to host RAM via PCIe DMA (`cuMemHostRegister(DEVICEMAP)`), not UVM page faults.

Tested: Qwen2.5-14B FP16 (28 GB) on Quadro RTX 5000 (16 GB VRAM), ~14 GB overflow to host RAM, ~0.73 tok/s on PCIe 3.0 x16.

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

## System Requirements

| Requirement | Details |
|-------------|---------|
| OS | Ubuntu 22.04 / 24.04 / 25.10 |
| Driver | NVIDIA **proprietary** 535+ (not open kernel module on Turing) |
| RAM | > model size in FP16 |
| Kernel headers | For GreenBoost module build |
| `ulimit -l` | `unlimited` (setup script configures this) |

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
