# C4A throughput and cost projection

The billing input is the user's console-derived estimate, **$70 / 730 h = $0.0959/hour**. It is not an audited GCP billing rate and excludes disk, network, taxes, and interruption effects.

For the selected stable eager configuration (12 intra-op threads, one inter-op thread, microbatch/effective batch 8), the 40-step confirmation measured 7,688.64 tokens/s. That projects to 27.679 million tokens/hour, **3.613 hours/100M tokens**, and **$0.3465/100M tokens** of VM runtime at the approximate rate.

The shorter frozen-batch sweep projected 14.480, 23.304, 27.953, and 30.281 million tokens/hour at 4, 8, 12, and 16 threads respectively. Corresponding approximate hours/100M are 6.906, 4.291, 3.578, and 3.302; approximate VM costs are $0.662, $0.412, $0.343, and $0.317. The 16-thread long sample had substantially worse tail variance and mean throughput than its short median suggests, which is why it was not selected.
