# LatticeLM final narration

Target: 298 seconds. Fish Audio / Ethan. Scene IDs map to replacement files named `<Scene>.wav` or `<Scene>.mp3`. Pauses are editorial targets; recorded duration takes priority.

## opening_01 · Opening · 0–8s

LatticeLM explores how architecture and training shape a compact language model. Watch its structure form as we inspect every parameter.

Expected pause: ~0.5s.

## architecture_01 · Architecture · 8–96s

The finished BASE model contains 48,636,168 trainable parameters. The interesting result is their allocation. Twelve Co4 causal blocks operate at width 552, with a 1,728-wide feed-forward path. The vocabulary has 4,096 entries, the context spans 256 tokens, and input and output embeddings are untied. Within each block, learned receptive latents meet the current query, key, and value contexts. MOD takes the receptive value squared, adds twice that value, adds context scaled by one plus its absolute magnitude, then applies ReLU6. The equation is an actual operation in the model: watch the three streams enter, transform, and return to the attention path. Query and key states then receive per-head RMS normalization. Their lengths change while their directions stay meaningful. Rotary position information follows before causal attention. Six query heads share only two physical key and value streams, three queries per group. This is real grouped-query attention, reducing key and value projection work. The hidden state widens inside each feed-forward block and returns to the 552-dimensional residual stream. Across twelve layers, that repeated geometry accounts for most of the parameter budget. Feed-forward matrices dominate the allocation, followed by attention projections, embeddings, and a small set of latents and norms. The design spends capacity on transformations that recur at every token. The small latent and normalization allocation matters because it shapes those repeated operations without consuming the majority of model capacity.

Expected pause: ~1.0s.

## training_01 · Training · 96–171s

Training begins with the DATA-D-v4 pipeline. A source mixture passes through normalization, deduplication, identifier handling, and tokenization. The certified stream contains 2,259,629,459 usable tokens. Optimization follows parameter geometry. Muon updates 44,089,344 two-dimensional projection parameters, or 90.65 percent of the model. AdamW handles the remaining 4,546,824 parameters, including embeddings, output weights, norms, and latents. The learning-rate schedule uses two percent warmup, 83 percent stable training, and 15 percent cosine decay. Follow the cursor across that schedule and the validation trajectory below it. At about 1.5 billion tokens, validation loss is 2.9145. By 1.9 billion tokens, after the late decay, it reaches 2.6305. That change is large enough to deserve attention: the final cooldown continued to improve the model after the long stable phase. The selected BASE checkpoint is the 1.9-billion-token state. Its official evaluation gives WikiText perplexity 21.9298 and bits per byte 1.413462. HellaSwag is 0.27096, ARC-Easy 0.36785, PIQA 0.56692, and WinoGrande 0.50987. Those values describe the pretraining checkpoint and remain separate from later post-training proxy screens. The final validation checkpoint is supported by these official measurements, rather than by the targeted reasoning screen.

Expected pause: ~3.75s.

## posttraining_01 · PostTraining · 171–216s

Post-training explored a branching method screen: supervised fine-tuning, optimizer and learning-rate choices, replay, ranking, joint specialization, and SimPO. RFT and RLVR stopped at capability gates when prerequisites were not met. One earlier, fixed 256-example screen showed the central tension clearly. Targeted reasoning accuracy rose sharply for a joint specialist, while natural-data loss worsened dramatically. The visual places reasoning behavior and retention on separate axes; motion toward one goal did not guarantee the other. This proxy measured a specialized behavior, not broad model capability. It motivated a recovery experiment and a fuller evaluation, where each candidate would face the same retention and broad-suite criteria.

Expected pause: ~6.38s.

## recovery_01 · Recovery · 216–261s

Could natural-data behavior return without erasing specialization? Ten million tokens of DATA-D causal training at constant three times ten to the minus five learning rate restored retention toward BASE while reducing the reasoning signal. A five-percent blend nearly tied BASE on GIBC: 0.4288 versus 0.4289. A seventeen-percent blend reached 0.3672 on the targeted proxy, yet GIBC fell to 0.4151. Mixed-objective follow-up used 500,000 tokens, generated reasoning completions, and 20 percent DATA-D replay. Its fifteen-percent blend reached 0.3418 proxy accuracy, with GIBC at 0.4154. The proxy captured real specialization, but gains did not predict broader GIBC improvement here. Future evaluation needs complementary measures of retention, targeted behavior, and broader capability.

Expected pause: ~3.75s.

## finalproof_01 · FinalProof · 261–273s

The selector starts with BASE, checks retention, then compares full GIBC. Evaluated recovery candidates reached that final comparison, but none cleared the improvement criterion. The supported final model remains BASE.

Expected pause: ~0.75s.

## recap_01 · Recap · 273–298s

LatticeLM ends with a 48.6-million-parameter, twelve-layer Co4 BASE model trained on 1.9 billion tokens. Real six-query, two-KV attention, QK-Norm, and a hybrid Muon plus AdamW optimizer define its core. The two, 83, fifteen WSD schedule finished at 2.6305 validation loss. Post-training revealed strong targeted specialization and substantially recoverable retention, while proxy gains failed to transfer to full GIBC. BASE stays selected under the final comparison.

Expected pause: ~0.62s.
