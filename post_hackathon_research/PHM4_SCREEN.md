# PHM-4 FFN screen

One PHM-4 FFN-only model was trained at a matched 16M parameter budget. The learned algebra uses four 4×4 matrices and four lower-dimensional transform blocks per linear map; attention and embeddings remain Co4. The 4096-wide PHM FFN reinvests parameter savings, so arithmetic is not reduced fourfold.

| Quantity | PHM-4 | Co4 control |
|---|---:|---:|
| Trainable parameters | 16,005,384 | 16,004,104 |
| Dense-equivalent FFN matrix shapes per layer | (8192×328), (328×4096) | (2048×328), (328×1024) |
| FFN matrix multiply-adds per token | 40,304,640 | 10,076,160 |
| Training tokens | 150,000,000 | 150,000,000 |
| Weighted held-out loss | 3.11432 | 3.09548 |
| WikiText-103 BPB | 1.64351 | 1.62941 |
| Training hours | 13.28 | 6.01 |
| 12-thread calibration tokens/s | 2669 | 5744 |
| 16-thread calibration tokens/s | 3188 | 6263 |
| Peak calibration RSS (GiB) | 8.34 | 5.27 |

PHM linear maps materialize dense-equivalent matrices; multiply-add work therefore tracks their 4096-wide FFN shapes rather than the number of free PHM parameters. A positive parameter result without a quality-per-wall signal is insufficient for promotion.
