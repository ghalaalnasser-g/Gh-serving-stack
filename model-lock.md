# Model lock (team record)

## The locked model

- Model id: Qwen/Qwen2.5-1.5B-Instruct-AWQ
- Quantisation: awq
- Why this one: Passed the smoke test at 10/10 (matching fp16 exactly), and
  the five-prompt spot check showed near-identical output to fp16 across all
  five prompts, in some cases word-for-word. VRAM read identical to fp16 on
  nvidia-smi (12,639 MiB both), since --gpu-memory-utilization 0.85 fills the
  same pool either way - the savings from smaller weights go to KV-cache
  capacity, not a lower memory reading.

## The launch flags
