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

## E002: data + 5 MiB baseline

E002 keeps the E001 architecture intact while adding a reproducible corpus-preparation path and a larger baseline configuration. The candidate is 1,317,280 parameters, approximately 6.58 MiB of FP32 weights with tied embeddings.

The initial corpus manifest targets TinyStories (roneneldan/TinyStories). The dataset card identifies it as English synthetic short stories and lists CDLA-Sharing-1.0; keep the raw corpus outside git and record the exact revision used for every experiment.

Typical flow:

1. Obtain the licensed source text locally.
2. Run the corpus preparation script with the source train/validation files.
3. Inspect the reported example and byte-token counts.
4. Measure the model with scripts/model_size.py.
5. Train with scripts/train_e002.py.

E002 is an English pretraining baseline. Multilingual training data is a separate experiment so that language coverage is not accidentally conflated with architecture scaling.

### E002 reproducible run

Install the optional data tooling:

```bash
python -m pip install -e ".[data,dev]"
```

Fetch the pinned TinyStories source and verify SHA-256:

```bash
python scripts/fetch_tinystories.py --split valid
python scripts/fetch_tinystories.py --split train
```

Then prepare the corpus while preserving the source train/validation split:

```bash
python scripts/prepare_corpus.py \
  --train data/raw/tinystories/TinyStories-train.txt \
  --val data/raw/tinystories/TinyStories-valid.txt \
  --out-dir data/processed/tinystories
```

Run the baseline training with validation monitoring:

```bash
python scripts/train_e002.py \
  --train data/processed/tinystories/train.txt \
  --val data/processed/tinystories/val.txt \
  --steps 5000 \
  --out checkpoints/e002.pt
```

Evaluate the saved checkpoint:

```bash
python scripts/evaluate_e002.py \
  --checkpoint checkpoints/e002.pt \
  --val data/processed/tinystories/val.txt
```

**Training note:** E002 training now uses a disk-backed memory map, so the full multi-GB corpus is not loaded into RAM. The full TinyStories run is not executed in GitHub Actions and should be benchmarked locally before committing to the complete 5,000-step run. The model target itself remains only ~6.58 MiB of FP32 weights.


## E003.3: Dolly instruction fine-tuning

E003 and E003.2 used a very small authored chat corpus and produced poor instruction-following generations. E003.3 switches to the human-generated Databricks Dolly 15K instruction dataset.

The source contains 15,015 English instruction-following records across multiple categories and is licensed CC BY-SA 3.0. The experiment uses a pinned source revision and SHA-256 checksum for reproducibility.

Only short, context-safe examples are selected for the 256-byte-token context window. The preparation script keeps category diversity, deduplicates examples, and writes a local JSONL train/validation split.

Download and verify the source:

```bash
python scripts/fetch_dolly.py
```

Prepare the compact E003.3 subset:

```bash
python scripts/prepare_e003_3.py \
  --source data/raw/dolly/databricks-dolly-15k.jsonl \
  --max-train 1600 \
  --max-val 240 \
  --out-dir data/processed/e003_3
```

Run the first measured fine-tune:

```bash
python scripts/train_e003_3.py \
  --base checkpoints/e002.pt \
  --train data/processed/e003_3/train.jsonl \
  --val data/processed/e003_3/val.jsonl \
  --steps 200 \
  --batch-size 8 \
  --out checkpoints/e003_3.pt \
  --best-out checkpoints/e003_3_best.pt
```

Evaluate:

```bash
python scripts/evaluate_e003.py \
  --checkpoint checkpoints/e003_3_best.pt \
  --max-new-tokens 60 \
  --temperature 0.1 \
  --top-k 1
```

E003.3 adds EOS to every response target and stops generation when EOS is emitted. The model architecture and tokenizer vocabulary size remain unchanged.
