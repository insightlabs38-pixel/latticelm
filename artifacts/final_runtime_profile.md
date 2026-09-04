# Phase 7B final-workload runtime profile

## Measured workload

The eager canonical run used frozen Co4-S, batch 8, context 128, and two PyTorch CPU threads. Through 1,000,448 tokens it sustained **2,068.31 tokens/s**, took **483.704 seconds**, and peaked at **3,047,112,704 bytes RSS**. No divergence occurred.

A separate whole-step pilot measured eager at 0.526001 seconds/step (1,946.76 tokens/s) and `torch.compile` at 0.481520 seconds/step (2,126.60 tokens/s) after two warmups over five timed steps. The compiler path was not introduced into the lineage.

## Kernel priorities

Prior Phase 6 attribution remains controlling: GEMMs and SDPA use optimized library primitives and should not be rewritten. The first custom-kernel preparation target remains the Co4 MOD/context elementwise forward/backward chain, followed by residual + RMSNorm. Output/loss fusion remains third and requires end-to-end evidence. No Triton kernel or end-to-end custom-kernel speedup is claimed in Phase 7B.
