# Co4-L provisional final-candidate profile

Profiled a complete FP32 training step on the frozen architecture shape (weights do not affect the operator graph), batch 8, context 128, two PyTorch threads. Three measured steps followed two warmups. Mean end-to-end step time was **1.0947 s**. `torch.profiler` included forward, cross-entropy, backward, and AdamW; the data pipeline was excluded because an in-memory batch was reused. Linux perf was not rerun; prior environment output remains preserved.

```text
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                               aten::mm        48.63%        1.417s        48.64%        1.417s       3.841ms     856.59 MB     856.59 MB           369
                                              aten::mul         9.21%     268.454ms         9.41%     274.150ms     134.387us       1.57 GB       1.57 GB          2040
aten::_scaled_dot_product_flash_attention_for_cpu_ba...         5.08%     148.094ms         5.33%     155.276ms       5.176ms     112.50 MB      -1.88 MB            30
                                            aten::fill_         3.71%     108.166ms         3.71%     108.166ms     166.922us           0 B           0 B           648
                       aten::_log_softmax_backward_data         3.54%     103.200ms         3.54%     103.200ms      34.400ms      48.00 MB      48.00 MB             3
      aten::_scaled_dot_product_flash_attention_for_cpu         2.90%      84.609ms         2.95%      85.898ms       2.863ms      37.97 MB      -1.54 MB            30
                                     aten::_log_softmax         2.68%      78.193ms         2.68%      78.193ms      26.064ms      48.00 MB      48.00 MB             3
                                            aten::copy_         2.65%      77.278ms         2.65%      77.278ms      25.993us           0 B           0 B          2973
                                              aten::add         2.12%      61.634ms         2.16%      62.802ms     138.635us     225.47 MB     225.47 MB           453
                                              aten::cat         1.62%      47.297ms         1.66%      48.301ms     805.023us     337.50 MB     337.50 MB            60
                                             aten::silu         1.36%      39.705ms         1.36%      39.705ms       1.323ms     112.50 MB     112.50 MB            30
                                             aten::sqrt         1.15%      33.526ms         1.15%      33.526ms     120.164us     182.53 MB     182.53 MB           279
                                              aten::div         1.10%      32.071ms         1.23%      35.781ms      89.006us     261.29 MB     261.29 MB           402
                                             aten::add_         1.06%      30.911ms         1.21%      35.182ms      33.797us           0 B      -2.18 KB          1041
                                              aten::sum         1.06%      30.894ms         1.09%      31.757ms      86.061us     555.75 KB     555.75 KB           369
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 2.914s

```

The dominant measured kernels are matrix multiplication/addmm, elementwise optimizer work, and SDPA/backward. At module level these map chiefly to Co4 QKV/output, FFN, output head/loss, their backward passes, and AdamW. The first fusion investigation should target the Co4 MOD elementwise chain around QKV only after separating it from GEMM with record functions. Custom GEMM is not recommended; output-head/loss fusion and residual+RMSNorm are secondary candidates. Peak operator allocations in the table are profiler allocations, not process RSS. No Triton speedup is claimed.
