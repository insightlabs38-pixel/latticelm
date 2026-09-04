# Triton-CPU AArch64 status

**Status: WORKING for minimal CPU-kernel correctness; not integrated into training.**

The official experimental `triton-lang/triton-cpu` source was cloned at commit `9a3dd8096b3c5b89a6dfeba012221f3fed450eb0`. Its pinned LLVM revision is `62b7cf9623fc310525f39ed69aaecc318a909731` (prebuilt Ubuntu ARM64 build 2), with Clang 18.1.3, LLD 18.1.3, CMake 3.31.10 from the isolated build environment, Ninja 1.13.2, Python 3.12.3, and CPU-only PyTorch 2.14.0. SLEEF is pinned at `93f04d869471ce4d007abaebb8c6a7bc62749f61`.

Two failed setup attempts are retained as negative results: first, `-fuse-ld=lld` failed because LLD was absent; second, the CPU CMake configuration failed because the shallow clone lacked SLEEF. Installing `lld` and initializing the exact SLEEF submodule resolved both. The editable AArch64 wheel then built successfully as Triton 3.8.0+git9a3dd809.

The bounded `scripts/smoke_triton_cpu.py` vector-add kernel compiled and returned exact equality for 257 FP32 elements (`sum=65792.0`). The upstream vector-add tutorial was stopped after three minutes because its broad autotuning/benchmark path exceeded the bounded smoke-test budget; this is recorded rather than treated as a correctness failure.

The optional Co4 MOD kernel was deferred. A production training replacement requires a custom backward, loss-trajectory match, and full-step speedup; implementing those after basic bring-up would risk delaying the canonical run. Best current MOD implementation: eager PyTorch reference.
