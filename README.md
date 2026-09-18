# Tiny-AI

A research project for building a genuinely generative, offline language model with a **5–10 MB model-weight target**.

## v0.1 / Experiment E001

E001 is the end-to-end baseline:

`text → byte tokenizer → tiny decoder Transformer → next-token prediction → generation`

Goals:
- keep the model architecture simple and inspectable
- make experiments reproducible
- measure parameter count, weight size, validation loss and generation speed
- establish a baseline before increasing model capacity

### Current baseline

- Python 3.11+
- PyTorch
- byte-level tokenizer (256 byte values + 3 special tokens)
- decoder-only Transformer
- context length: 128 tokens
- 2 layers / 4 attention heads / 128 hidden size
- tied input/output embeddings
- causal next-token prediction

E001 is intentionally much smaller than the eventual 5–10 MB target. It exists to prove the complete pipeline first.

## Repository layout

```
tiny-ai/
├── configs/
├── data/
├── evaluation/
├── inference/
├── model/
├── scripts/
├── tests/
└── tokenizer/
```

## Quick start

Install:

```bash
python -m pip install -e .
```

Run the architecture smoke test:

```bash
python scripts/smoke_test.py
```

Train on a small local text file:

```bash
python scripts/train_e001.py --text data/example.txt --steps 500
```

Generate text from a checkpoint:

```bash
python scripts/chat.py --checkpoint checkpoints/e001.pt
```

> Training data is not included yet. E001's example text is only a pipeline smoke-test corpus, not a meaningful training dataset.

## Experiment roadmap

- E001 — pipeline baseline
- E002 — larger capacity + curated pretraining data
- E003 — ~5 MB class model
- E004 — ~8–10 MB candidate
- Quantization experiments — INT8/INT4 after a strong FP baseline

## Principles

1. No scraped dataset without checking licensing.
2. Every dataset gets a manifest describing source, license and processing.
3. Every model experiment gets a config and measurable evaluation.
4. Never claim capability from model size alone.
5. Keep the model itself separate from optional external knowledge/retrieval.
