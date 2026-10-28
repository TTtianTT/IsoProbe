# Local model inventory (2026-10-28)

Sources checked: `hf cache list` for `HF_HOME=/dataset1/zailong/cache/huggingface` and `~/.cache/huggingface`
(neither holds any model weights; only a TurnPrint dataset and LoRA adapter stubs), model paths referenced by the
TurnPrint configs, and every directory under `/dataset1/zailong` with a `config.json`
(all of them live under `/dataset1/zailong/models`).
Parameter counts come from the safetensors headers (GPT-2 family: from `model.parameters()`, since those checkpoints also store the causal-mask buffers). "Post-trained" = instruct/chat or reasoning model, not a base checkpoint.

## Selected → `models.txt` (by parameter count)

| model | local path | params | layers | d | variant | BOS added by tokenizer |
|---|---|---|---|---|---|---|
| gpt2 (small) | /dataset1/zailong/models/gpt2 | 0.124B | 12 | 768 | base (sanity check; downloaded 2026-10-28) | no |
| gpt2-medium | /dataset1/zailong/models/gpt2-medium | 0.355B | 24 | 1024 | base (downloaded 2026-10-28, round 2) | no |
| gpt2-large | /dataset1/zailong/models/gpt2-large | 0.774B | 36 | 1280 | base (downloaded 2026-10-28, round 2) | no |
| gpt2-xl | /dataset1/zailong/models/gpt2-xl | 1.558B | 48 | 1600 | base (downloaded 2026-10-28, round 2) | no |
| Qwen2.5-7B-Instruct | /dataset1/zailong/models/Qwen2.5-7B-Instruct | 7.616B | 28 | 3584 | **instruct** | no |
| Llama-3.1-8B | /dataset1/zailong/models/Llama-3.1-8B | 8.030B | 32 | 4096 | **base** (downloaded 2026-10-28, round 2) | yes (`<\|begin_of_text\|>`) |
| Llama-3.1-8B-Instruct | /dataset1/zailong/models/Llama-3.1-8B-Instruct | 8.030B | 32 | 4096 | **instruct** | yes (`<\|begin_of_text\|>`) |
| Qwen3-8B-Base | /dataset1/zailong/models/Qwen3-8B-Base | 8.191B | 36 | 4096 | **base** (downloaded 2026-10-28, round 2) | no |
| Qwen3-8B | /dataset1/zailong/models/Qwen3-8B | 8.191B | 36 | 4096 | **post-trained** (hybrid thinking; base is Qwen3-8B-Base) | no |
| Mistral-Small-24B-Instruct-2501 | /dataset1/zailong/models/Mistral-Small-24B-Instruct-2501 | 23.572B | 40 | 5120 | **instruct** | yes (`<s>`) |
| Olmo-3.1-32B-Instruct | /dataset1/zailong/models/Olmo-3.1-32B-Instruct | 32.234B | 64 | 5120 | **instruct** | no (first token is text) |
| Qwen3-32B | /dataset1/zailong/models/Qwen3-32B | 32.762B | 64 | 5120 | **post-trained** (hybrid thinking) | no |

Round 1 used only models that were already local, and none of them was a modern **base** model.
Round 2 (2026-10-28) downloaded gpt2-medium/large/xl (to locate the scale boundary) and two base models,
Llama-3.1-8B and Qwen3-8B-Base, each paired with a post-trained model of the same architecture already in the set.
None of the modern models ties input/output embeddings (`tie_word_embeddings=False`); the GPT-2 family does.
The smallest modern model (used for the step-3 smoke test instead of Qwen2.5-0.5B) is Qwen2.5-7B-Instruct.

## Excluded

| path | reason |
|---|---|
| gemma-3-12b-it, gemma-3-27b-it | multimodal (`Gemma3ForConditionalGeneration`, has `vision_config`) |
| Qwen3.5-4B, MIMESIS-9B | multimodal (`Qwen3_5ForConditionalGeneration`, has `vision_config`) |
| GLM-4.5-Air-FP8 | decoder-only text MoE (~106B), but weights are compressed-tensors FP8; the protocol prescribes a bf16 forward, and quantization could itself alter the geometry. Can be added later if wanted |
| Meta-Llama-3-8B-Instruct | incomplete download (only LICENSE / README, no config or weights) |
| LLM2Vec-Meta-Llama-31-8B-Instruct-mntp(-supervised) | embedding model (bidirectional encoder adapter) |
| bge-large-en-v1.5 | embedding model (BERT) |
| roberta-large-mnli | encoder classifier |
| reference-frame-cross-backbone/* (flan-t5-*, t5gemma-*) | encoder–decoder |
| spectral-surgery/* | LoRA adapters only, no standalone weights |

Note: Mistral-Small-24B ships both `consolidated.safetensors` (Mistral format) and HF shards; transformers loads the HF shards via the index.
