# CPU training context

LatticeLM was not designed around a record claim. The CPU constraint emerged from the available compute environment, and the production run was completed because the architecture, data pipeline, checkpointing system, and training implementation made that constraint workable. After the final run was complete, its combination of model size, token count, hardware, and training method appeared unusual enough to justify a separate comparison against publicly documented CPU-trained language models.

This note records that comparison as a dated literature and repository search rather than as an unconditional world-record assertion. The search was performed on **October 1, 2026** across published papers, public repositories, and model cards, with particular attention to historical RNN language models, modern causal language models, CPU-only training projects, and methods that advertise unusually large CPU-trained parameter counts. The result should therefore be read as a claim about the public evidence located under the criteria below; unpublished runs, inaccessible internal projects, or poorly documented counterexamples may exist.

## LatticeLM production run

The selected LatticeLM model contains **48,636,168 trainable parameters** and was trained from random initialization for **1.9 billion tokens** on a single Google Cloud `c4a-standard-16` instance. The host provides 16 Google Axion CPU cores and 64 GB of RAM, and no GPU or TPU was used for the production pretraining run.

The recorded wall-clock duration was **7.73525463 days, or 185.646 hours**, corresponding to an average whole-run throughput of approximately **2,842.9 tokens per second**. Process CPU utilization remained around **1,550%**, which corresponds to approximately 15.5 of the 16 cores being occupied on average, or about **96.9% aggregate core utilization**. Using the conventional `6NT` approximation, which is a close approximation for the production Co4 configuration, the final pretraining run represents approximately **5.54 × 10^17 FLOPs, or 0.554 EFLOP**. This compute estimate refers only to the final production pretraining run and does not include data preparation, evaluation, optimizer overhead outside the approximation, or the separate research runs that preceded it.

## Comparison definition

A broad statement such as “largest language model trained on CPU” is not meaningful enough for this comparison. Historical language-model systems reached nominal parameter counts in the billions on CPUs by relying on hierarchical, sampled, class-based, or sparse output computation, while modern CPU systems can reach still larger nominal sizes through dynamic sparsity or conditional activation. Those are legitimate models and important systems results, but their parameter counts do not describe the same per-token training computation as LatticeLM.

For this reason, the comparison asks a narrower question: whether a larger publicly documented model could be found that satisfies all of the following conditions at the same time.

1. The model is an autoregressive or causal neural language model rather than a conventional n-gram system or a non-language-model network.
2. The reported model is trained from random initialization rather than beginning from a GPU- or accelerator-trained checkpoint.
3. The run performs at least **one billion training tokens or token positions**, so the comparison represents substantial pretraining rather than CPU compatibility, a smoke test, or a short ablation.
4. The complete pretraining run is performed on **one CPU-only physical node**, without GPU or TPU compute and without distributing the model or training stream across a CPU cluster.
5. Training uses the model's conventional full next-token computation rather than dynamic neuron sparsity, conditional expert routing, sampled softmax, hierarchical softmax, class-based output approximation, or another mechanism whose purpose is to avoid evaluating or updating a large fraction of the nominal model for each token.

The fifth condition does not treat ordinary embedding lookup as sparse training. Token-indexed embeddings are part of standard language-model computation. The exclusion concerns architectural or objective-level mechanisms that make nominal parameter count diverge substantially from the parameters involved in the normal training computation.

## Historical and modern counterexamples

The search found several CPU-trained language models with much larger nominal parameter counts than LatticeLM. They are important counterexamples to any broad CPU-training record claim, but they do not satisfy the narrower comparison above.

| Work | Reported scale | CPU training evidence | Why it is not the same comparison |
|---|---|---|---|
| **LatticeLM** | **48.64M parameters, 1.9B tokens** | One 16-core Google Axion node; 185.646 h | Reference case: random initialization, single-node CPU-only pretraining, conventional full-model next-token training |
| **Chelba et al., One Billion Word Benchmark RNN + MaxEnt** | Up to **20B nominal parameters** on almost 1B words | The paper reports large RNN/MaxEnt configurations on a single standard CPU machine without GPUs | The very large count includes MaxEnt/n-gram machinery and the work explicitly uses techniques such as hierarchical or class-based output computation to make the large vocabulary tractable; nominal parameter count is therefore not comparable to conventional dense decoder training |
| **BlackOut RNNLM** | **Billions of parameters**, one billion words | The paper reports 1–10 day training on a single CPU machine | BlackOut is explicitly a sampling-based approximation: only a sampled subset of the output layer participates in each training batch, with `K` typically a small fraction of the vocabulary |
| **ThirdAI BOLT2.5B** | **2.5B parameters**, approximately 40B tokens as reported by ThirdAI | CPU-only pretraining on multiple dual-socket Sapphire Rapids systems | BOLT is built around dynamic sparse algorithms that activate only a subset of neurons for an input, specifically avoiding full dense computation; the reported pretraining also uses multiple CPU systems rather than one 16-core node |
| **InftyThink** | Approximately **60M parameters** | Repository explicitly forces the JAX CPU backend | The published configuration runs 10,000 steps with effective batch 32 and maximum sequence length 1,024, giving an upper bound of 327.68M token positions even if every sequence is full length; it does not meet the ≥1B-token substantial-pretraining threshold |
| **VihaanNet10x-50M** | Approximately **50M parameters** | Model card reports training from scratch on Colab CPU | The card reports WikiText-2 at approximately 2M tokens for three epochs, several orders of magnitude below LatticeLM's 1.9B-token production run |

### BlackOut

[BlackOut](https://arxiv.org/abs/1511.06909) is the strongest historical reason not to use an unqualified “largest CPU-trained model” statement. The paper reports billion-parameter recurrent language models trained on one billion words in 1–10 days on a single CPU machine. However, this result is made possible by a sampling-based approximation to the large output layer. The paper states that a subset of the output layer is sampled and trained for each batch; in its notation, the sampled set can be roughly `V/200`, rather than evaluating and updating the entire million-word output layer for every target. The full output distribution is restored for evaluation.

That is a substantial and legitimate CPU-training result, but it answers a different systems question from LatticeLM. BlackOut increases nominal model size by avoiding most of the otherwise dominant output computation during training, whereas LatticeLM's parameter count describes the ordinary model used throughout its forward and backward passes.

### One Billion Word Benchmark RNN systems

The [One Billion Word Benchmark](https://arxiv.org/abs/1312.3005) provides an earlier family of large CPU language-model results. Its RNN + MaxEnt systems report nominal parameter counts as high as 20 billion and training times measured in days on a single standard CPU machine. These systems combine recurrent neural components with very large MaxEnt/n-gram components and use output-side techniques designed to reduce the cost of large vocabularies. The same work discusses hierarchical softmax, where only the branches relevant to a target need to be evaluated instead of the complete output vocabulary.

These results demonstrate that very large nominal CPU language models existed well before modern Transformers, but they also show why nominal parameter count alone is not a useful comparator. A model can contain billions of stored parameters while arranging training so that only a small structured subset is touched for each example.

### ThirdAI BOLT

ThirdAI's [BOLT2.5B announcement](https://medium.com/thirdai-blog/introducing-the-worlds-first-generative-llm-pre-trained-only-on-cpus-meet-thirdai-s-bolt2-5b-10c0600e1af4) reports a 2.5-billion-parameter generative model pretrained without GPUs on Sapphire Rapids CPU systems. ThirdAI's description of BOLT through [AWS](https://aws.amazon.com/blogs/machine-learning/accelerating-large-scale-neural-network-training-on-cpus-with-thirdai-and-aws-graviton/) explains the central mechanism: proprietary dynamic sparse algorithms select only a subset of neurons for a given input and thereby avoid full dense computation. The BOLT2.5B run also used multiple high-core-count CPU systems rather than a single modest node.

BOLT therefore establishes that CPU-only pretraining can reach far beyond LatticeLM in nominal parameter count when the system is explicitly designed around sparse activation and distributed CPU resources. It does not provide a larger example under the single-node, conventional-full-computation definition used here.

### Recent CPU-only projects near LatticeLM's scale

The public search also found models much closer to LatticeLM in architecture and scale. [InftyThink](https://github.com/jadenfix/InftyThink) documents a decoder-only Transformer at approximately 60M parameters and explicitly forces all experiments onto the JAX CPU backend. Its published base configuration, however, runs 10,000 steps with effective batch size 32 and maximum sequence length 1,024 on a 5,000-example math dataset. Even treating every training sequence as full length gives an upper bound of 327.68M token positions, so it demonstrates CPU feasibility at a slightly larger parameter count without representing a billion-token pretraining campaign.

[VihaanNet10x-50M](https://huggingface.co/vihaan134354/VihaanNet10x-50M) similarly reports a roughly 50M-parameter custom language model trained from scratch on CPU, but the model card gives a training corpus of approximately 2M WikiText-2 tokens for three epochs. This is useful evidence that models around LatticeLM's parameter scale can be trained on CPUs, while also illustrating the difference between proving that training is possible and sustaining that training for billions of tokens.

## Search result as of October 1, 2026

Under the definition above, the search found **no larger publicly documented causal language model than LatticeLM's 48,636,168 trainable parameters that completed at least one billion tokens of pretraining from random initialization entirely on a single CPU-only node using conventional full-model training, without dynamic sparsity, MoE routing, or sampled, hierarchical, or class-based output approximations**.

This statement is intentionally narrower than a world-record claim. It describes the result of a targeted search of published literature, public model cards, and open-source repositories as of October 1, 2026. It does not prove that no qualifying private, unpublished, deleted, or poorly indexed run exists. A newly discovered public counterexample should therefore update this document rather than be treated as contradicting a permanent record assertion.

The narrower result is still useful context for LatticeLM because the conditions describe the actual resource constraint of the production run rather than being chosen only to exclude inconvenient comparisons. The model was initialized from scratch, trained through 1.9B tokens, kept on one 16-core CPU node for the complete run, and did not use conditional sparsity or an approximate large-vocabulary objective to reduce the normal next-token computation. Its 7.735-day wall time is a direct consequence of carrying that conventional training workload to completion on a modest CPU host.

## Interpretation

The comparison should not be read as evidence that CPU training is generally more efficient than accelerator training. Modern GPUs and TPUs provide dramatically higher throughput for this workload, which is precisely why serious from-scratch CPU pretraining has become uncommon. LatticeLM instead demonstrates that a substantial modern language-model training campaign can still be completed on a single general-purpose CPU node when model scale, implementation efficiency, data flow, checkpointing, and experiment design are built around that constraint.

That distinction is the useful result: not that LatticeLM is the largest model ever associated with CPU training, but that a nearly 49M-parameter causal model was taken through a full 1.9B-token pretraining run on one 16-core CPU machine without relying on accelerator compute, distributed CPU hardware, or sparse-training shortcuts. The historical examples above make that statement more precise rather than less significant.
