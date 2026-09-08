# Systems screen

- fp32_eager: 5210.75 tok/s, +0.00%, RSS 1188433920 bytes, PASS, DROP
- fp32_compile: 5901.06 tok/s, +13.25%, RSS 1201586176 bytes, PASS, CANONICAL BACKEND CANDIDATE
- bf16_eager: 3177.67 tok/s, -39.02%, RSS 1078042624 bytes, PASS, DROP
- bf16_compile: 3361.83 tok/s, -35.48%, RSS 1084387328 bytes, PASS, DROP

BEST VERIFIED BACKEND: fp32_compile
BEST SPEEDUP: 13.25%
TRITON TESTED: NO
TRITON RESULT: gated off unless profiling threshold is independently met; corpus takes priority
SCHEDULEFREE+ TESTED: NO
SCHEDULEFREE+ RESULT: deferred pending corpus completion and four-hour slack
