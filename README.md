# Tiny-AI

A research project for building a genuinely generative, offline language model with a **5–10 MB model-weight target**.

## Current experiments

- **E001** — end-to-end pipeline baseline.
- **E002** — TinyStories language-model baseline.
- **E003** — narrow chat/instruction fine-tuning on top of E002.
- **E003.1** — expanded authored chat data + context-safe story sampling.
- **E004** — later experiment based on measured E003.1 results.

## E002 baseline

E002 uses:
- byte-level tokenizer (256 byte values + 3 special tokens)
- decoder-only Transformer
- context length: 256 tokens
- 4 layers / 4 attention heads / 160 hidden size
- tied input/output embeddings

Measured E002 checkpoint size:
- parameters: **1,317,280**
- FP32 weights: **5.025 MiB**
- FP16 weights: **2.513 MiB**
- INT8 weights: **1.256 MiB**

The model was trained on the official TinyStories train split and evaluated on the official validation split. Training uses a disk-backed memory map so the full multi-GB corpus is not loaded into RAM.

## E003: chat/instruction fine-tuning

E002 is a base language model: it continues text, but it was not trained to answer a user.

E003 keeps the same tokenizer and architecture and fine-tunes the E002 checkpoint on a small JSONL instruction corpus built locally from:
1. selected TinyStories examples wrapped as story requests, and
2. authored greeting/small-talk examples.

The loss is applied to **assistant response tokens only**, so the model is not rewarded for memorizing the user prompt as the target.

Prepare E003 data:

```bash
python scripts/prepare_e003.py \
  --source data/processed/tinystories/train.txt \
  --n-stories 1000 \
  --out-dir data/processed/e003
```

Run a cheap smoke experiment first:

```bash
python scripts/train_e003.py \
  --base checkpoints/e002.pt \
  --train data/processed/e003/train.jsonl \
  --val data/processed/e003/val.jsonl \
  --steps 50 \
  --batch-size 8 \
  --out checkpoints/e003_smoke.pt
```

Evaluate:

```bash
python scripts/evaluate_e003.py \
  --checkpoint checkpoints/e003_smoke.pt
```

Then run the first real E003 experiment:

```bash
python scripts/train_e003.py \
  --base checkpoints/e002.pt \
  --train data/processed/e003/train.jsonl \
  --val data/processed/e003/val.jsonl \
  --steps 1000 \
  --batch-size 8 \
  --out checkpoints/e003.pt
```

E003 is deliberately narrow. It is an experiment in instruction following and conversation formatting, not a claim of general knowledge or a general-purpose assistant.


### E003.1: first stronger chat experiment

E003.1 keeps the E002 architecture and checkpoint size unchanged, but improves the training set:
- story examples are sampled only when they fit the 256-byte-token context
- the authored conversational set is expanded substantially
- response-only loss remains unchanged

Prepare the data:

```bash
python scripts/prepare_e003.py \
  --source data/processed/tinystories/train.txt \
  --n-stories 500 \
  --out-dir data/processed/e003
```

Run the first real E003.1 experiment:

```bash
python scripts/train_e003.py \
  --base checkpoints/e002.pt \
  --train data/processed/e003/train.jsonl \
  --val data/processed/e003/val.jsonl \
  --steps 300 \
  --batch-size 8 \
  --out checkpoints/e003_1.pt
```

Evaluate:

```bash
python scripts/evaluate_e003.py \
  --checkpoint checkpoints/e003_1.pt \
  --max-new-tokens 80
```

The 300-step run is intentionally a measured first experiment; increase training only after checking chat quality and validation behaviour.
