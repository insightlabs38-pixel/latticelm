# LatticeLM cloud system report

Captured 2026-09-01 UTC. The complete command output is preserved in
`artifacts/logs/cloud_system_raw.txt`; the structured source of truth is
`artifacts/cloud_environment.json`.

## Compute allocation

- Host-reported CPU: Intel Xeon Platinum 8370C at 2.80 GHz (`GenuineIntel`),
  under KVM virtualization.
- The container sees three logical CPUs in one NUMA node, while cgroup quota is
  `200000 100000` (two CPU-equivalents). The effective cpuset is `0-2`.
  Consequently, benchmarks include 1/2/3/4-thread requests, but 3 and 4 threads
  can be quota-throttled and 4 is oversubscribed.
- Topology reported by `lscpu`: one socket, three cores/socket, one thread/core.
  Cache: 96 KiB L1d and 64 KiB L1i (two instances), 2.5 MiB L2 (two
  instances), and 48 MiB shared L3.
- ISA exposed to the guest: AVX, AVX2, FMA, and AVX-512 F/CD/DQ/BW/VL. Neither
  VNNI, AVX-512 BF16, nor AMX is exposed. No low-precision instruction claim is
  therefore made.

## OS, memory, and storage

- Ubuntu 24.04.4 LTS, Linux 6.18.35, x86-64.
- `/proc/meminfo` reports 18,802,476 KiB, but the cgroup hard limit is
  17,179,869,184 bytes (16 GiB); experiments must use the latter limit.
- The safe temporary-storage benchmark could not use `/usr/bin/time` because
  that executable is absent. No storage throughput number is reported.

## Python and PyTorch

- Python 3.14.4 (GCC 13.3.0), PyTorch 2.13.0+cu130. This wheel was built with
  MKL 2024.2, oneDNN 3.12, OpenMP, and an AVX-512 CPU dispatch path. CUDA being
  present in the build metadata does not imply that a GPU is available or used.
- PyTorch and Python both detect three CPUs; default intra-op and inter-op
  thread counts are three.
- Toolchain found: GCC/G++ 13.3, Clang 17.0, CMake 3.28.3, Ninja 1.11.1, and
  Git 2.43.0.

## CPU microbenchmarks

`scripts/cloud_benchmark.py` produced 49 cases in JSON and CSV. Every latency
is a median with standard deviation and interquartile dispersion, following
warmup. The quick capture used seven measured repeats.

- FP32 add triad: 2.275 ms median, 0.126 ms standard deviation, approximately
  42.20 GB/s effective bandwidth with three requested threads.
- RMSNorm-shaped reference (`8x128x256`): best observed median 0.158 ms at two
  threads.
- Causal SDPA (`B=2,H=4,T=64,D=64`): best observed median 0.081 ms at two
  threads. Lengths 64, 128, and 256 are all retained in the result files.
- Embedding gathers cover approximately 0.5M, 2M, 8M, and 16M table parameters
  with both random and locality-friendly IDs.
- GEMM/GEMV covers square 256 and 512 matrices, a 256x768 projection,
  sequence-flattened 1024x256 input, and small 1/128-row projections for every
  requested thread count.

These are host observations, not portable peak-performance claims. The CPU
quota makes repeated end-to-end measurements more useful than extrapolating
from a single microbenchmark.

## `perf`

`perf` is not installed, so hardware counters (cycles, instructions, IPC and
cache/branch events) could not be collected. The exact diagnostic is preserved
in `artifacts/logs/perf_stat.txt`.

## Triton-CPU preparation

The current official repository was cloned recursively and its current README
was read. The inspected commit was
`b27ed1ad89239b0643947e963bbab39f3664c07b`. It currently recommends a Python
3.12 `uv` environment, dependency installation, and editable source build,
then the CPU vector-add and SFC-matmul tutorials. This capture establishes that
the Linux compiler prerequisites exist; it does **not** claim that Triton-CPU
was built or smoke-tested during this bounded capture. The README snapshot and
toolchain versions are preserved under `artifacts/logs/` for the follow-up.

