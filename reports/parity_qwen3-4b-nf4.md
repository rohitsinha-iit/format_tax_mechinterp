# vLLM vs HF parity: qwen3-4b-nf4

- prompts: 20 MATH-500 problems x ('FF', 'JS') = 40
- top-1 agreement: 1.000 (threshold 0.90) -> PASS
- mean KL(vLLM||HF) over vLLM top-20 (renormalised): 0.00387; max 0.07190
- by condition: {'FF': 1.0, 'JS': 1.0}
