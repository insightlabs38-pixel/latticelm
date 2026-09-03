# Phase 6 non-GEMM Triton target analysis

## Scope and measured evidence

The existing complete-step Co4-L FP32 profile measured 1.0947 s/step after warmup with batch 8, context 128, and two CPU threads. GEMM (`aten::mm`) accounts for 48.63% of profiler self-CPU time and is deliberately excluded as a custom target because oneDNN/BLAS already owns that path. The remaining measured categories include `mul` 9.21%, SDPA backward 5.08%, `fill_` 3.71%, log-softmax backward 3.54%, SDPA forward 2.90%, log-softmax 2.68%, `copy_` 2.65%, `add` 2.12%, `cat` 1.62%, SiLU 1.36%, `sqrt` 1.15%, `div` 1.10%, `add_` 1.06%, and reductions 1.06%. Percentages are operator-wide attribution, not unsupported claims that all calls belong to Co4.

Inspection of `CausalCo4Attention` and the profiler identifies repeated Co4-specific MOD/context chains around projected Q/K/V: tensor splitting, sigmoid/tanh-style modulation, multiplication, concatenation, and residual combination. These chains create several full-sized intermediate tensors and participate in autograd, making memory traffic rather than arithmetic intensity the likely constraint.

## Ranked targets

1. **Co4 MOD/context elementwise chain, forward + backward.** Best first target. It aggregates a meaningful subset of `mul`, `add`, `cat`, `copy_`, and activation work; has a compact PyTorch correctness oracle; produces avoidable intermediates; and is unique to the selected family. Instrument it with explicit `record_function` ranges before kernel work so the exact share is separated from optimizer elementwise operations.
2. **Residual-add + RMSNorm fusion.** Repeated twice per layer and again at the output. It combines reductions, square/sqrt/div/mul and residual traffic, has a straightforward reference, and is architecture-stable. Backward and numerical tolerances must be tested.
3. **Output log-softmax / NLL fusion.** The paired forward/backward operators account for 6.22% self CPU before surrounding transformations and allocate large vocabulary-shaped tensors. A fused training loss can avoid materialization, but established framework kernels and numerical stability raise implementation risk.
4. **RoPE application and layout conversion.** `empty_like`, indexed writes, multiplication/addition, transpose/contiguous, and copies are fusible, but aggregate benefit is likely below the first three and CPU SDPA layout constraints may erase gains.
5. **AdamW elementwise update.** Aggregate `mul/add/sqrt/div` is substantial, but optimizer fusion is generic, state-heavy, and better served by framework `foreach`/fused paths before custom Triton.
6. **SDPA support operations.** Backward and forward are meaningful, but the core is already an optimized PyTorch CPU flash-attention kernel. Only surrounding repeat/layout operations are plausible targets; replacing SDPA itself is not recommended.

## Decision

The next phase should first add module-level ranges and benchmark a fused **Co4 MOD/context forward/backward** reference. Residual+RMSNorm is the fallback if measured MOD attribution is too small. No custom kernel was implemented, no GPU was available, and no Triton speedup is claimed.
