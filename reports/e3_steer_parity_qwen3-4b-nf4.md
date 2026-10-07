# E3 steered parity (vLLM hook vs HF hook): qwen3-4b-nf4

- L* = 22; 20 dev prompts per setting; next-token distribution at the last prompt position

| setting | n | top-1 agreement | mean KL (top-20, renormalised) |
|---|---|---|---|
| JS|minus_b | 20 | 1.000 | 0.00375 |
| FF|plus_b | 20 | 1.000 | 0.00002 |
| JS|none | 20 | 0.950 | 0.00618 |

- steered top-1 agreement 1.000 (threshold 0.90) → **PASS**
- vLLM hook token counts (steered vs prompt tokens submitted): {'JS|minus_b': {'steered_tokens': 3918, 'prompt_tokens': 3918}, 'FF|plus_b': {'steered_tokens': 2214, 'prompt_tokens': 2214}, 'JS|none': {'steered_tokens': 0, 'prompt_tokens': 3918}}
