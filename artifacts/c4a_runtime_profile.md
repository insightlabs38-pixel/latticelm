# C4A eager runtime profile

Five complete Co4-S forward/loss/backward/clip/AdamW steps were profiled at the selected 12-thread, batch-8, context-128 setting after three warmups. Total profiler self CPU time was 691.641 ms.

Matrix multiplication remained dominant at 46.96% self CPU. Co4/elementwise-relevant operations included `mul` 6.64%, `copy_` 3.72%, `add` 1.66%, `cat` 0.99%, `pow` 0.73%, and ReLU6 backward (`hardtanh_backward`) 0.52%. CPU flash-SDPA forward and backward accounted for 2.66% and 4.89%. SiLU forward/backward accounted for 1.06%/0.93%. Reductions/norm-related `sum`, `div`, vector norm, and `sqrt` contributed 3.47%, 1.00%, 1.41%, and 0.45%. Output loss log-softmax forward/backward contributed 0.57%/0.55%. AdamW's recorded range was 8.04% total CPU; zeroing gradients was 1.05% self CPU.

This closely resembles the previous x86 ranking—GEMM first and Co4 MOD/context elementwise traffic the best non-GEMM fusion candidate—but ARM has a lower measured log-softmax share and retains optimized CPU flash attention. The best current MOD implementation remains the eager PyTorch reference; no custom kernel is integrated.
