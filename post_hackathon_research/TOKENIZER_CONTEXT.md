# Tokenizer and context efficiency

The models use the same raw document IDs, source order, train/validation split, optimizer, batch semantics, and approximately 16M trainable parameters. Untied embedding/head costs are offset by FFN widths 1024, 752, and 205 for 4K, 8K, and 16K. The source sample is the pinned phase7f raw cache plus the pinned 40MB FineMath sample; this controlled subset is distinct from full DATA-D-v4, so its 4K arm is the proper direct comparator.

| Measure | 4K | 8K | 16K |
|---|---:|---:|---:|
| Trainable parameters | 16,004,104 | 16,014,600 | 16006072 |
| Mixture-weighted raw bytes/token | 3.2554 | 3.6425 | 3.9777 |
| Mixture-weighted characters/token | 3.2380 | 3.6230 | 3.9565 |
| Mixture-weighted words/token | 0.5393 | 0.6033 | 0.6586 |
| WikiText BPB at 150M tokenizer tokens | 1.64983 | 1.65349 | 1.65085 |
| 8K BPB near 4K's 150M-token raw-byte exposure | — | 1.73100 | — |

The byte-matched 8K checkpoint was captured at 134,062,080 tokenizer tokens; the target was 134,062,080. Raw-byte matching uses the mixture-weighted subset compression ratio, and is approximate because each source has a distinct ratio. Equal-token endpoints use 150M tokenizer tokens and process different byte volumes.

The 16K training gate: evidence and time gate passed. Its bytes/token gain over 8K was 9.20%; the 8K BPB change versus 4K was 0.22%.

The raw subset is smaller than full DATA-D-v4. Source streams repeat deterministically once their blocks are exhausted; per-source target exposure divided by available train tokens is recorded as `source_reuse_factors_at_150m` in the comparison JSON. Any quality gain therefore applies to this repeated common subset and needs larger-corpus confirmation.

The full eight-task prompt audit found a maximum meaningful truncation fraction of 23.90% at context 256. The registered context decision and any 512-token screen are in CONTEXT_512.md.

## Benchmark request lengths

Each cell gives mean prompt-plus-continuation tokens under the same full zero-shot task requests. The JSON evidence also records per-task p95, maxima, crossings at 128/256/512/1024, and actual left truncation under context 256.

| Task | 4K | 8K | 16K |
|---|---:|---:|---:|
| arc_challenge | 53.8 | 45.0 | 39.7 |
| arc_easy | 47.0 | 39.1 | 34.0 |
| boolq | 208.9 | 183.6 | 165.7 |
| hellaswag | 106.2 | 95.2 | 87.4 |
| lambada_openai | 102.7 | 93.5 | 87.1 |
| openbookqa | 20.4 | 18.2 | 16.5 |
| piqa | 53.9 | 46.2 | 40.5 |
| winogrande | 31.2 | 27.8 | 25.0 |

## Embedding parameter economics

The primary comparison keeps untied input and output matrices. As a secondary accounting exercise, tying them would save one vocabulary-by-328 matrix; no tied model is trained here.

| Vocabulary | Untied embedding/head parameters | Tied parameters | Savings if tied |
|---:|---:|---:|---:|
| 4,096 | 2,686,976 | 1,343,488 | 1,343,488 |
| 8,192 | 5,373,952 | 2,686,976 | 2,686,976 |
| 16,384 | 10,747,904 | 5,373,952 | 5,373,952 |

## Conditional 16K screen

The matched 16K model has 16,006,072 parameters, 3.9777 mixture-weighted bytes/token, and 1.65085 WikiText BPB at 150M tokenizer tokens.
Its byte-matched BPB was 1.73429 at 122,765,312 target tokens. Tying the 16K embedding/head would save 5,373,952 parameters; weight tying was not changed in the primary ablation.
