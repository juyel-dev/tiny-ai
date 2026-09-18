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


## E004: BPE model from scratch

E003.x showed that aggressively fine-tuning the 1.3M-parameter byte-level model did not produce useful instruction-following generations. E004 changes the representation and trains a new model from scratch.

Tokenizer:
- byte-level BPE
- 2,048 vocabulary entries
- `<pad>`, `<bos>`, `<eos>`, `<unk>`
- tokenizer training uses the first 100,000 TinyStories examples plus the full compact Dolly source
- the full training corpora remain disk-backed

Model:
- 4 Transformer blocks
- 6 attention heads
- 192 hidden size
- 256-token context
- tied input/output embeddings
- **2,219,136 parameters**
- **~8.465 MiB FP32 weights**

BPE training is done with the Hugging Face Tokenizers library; its BPE trainer supports a target vocabulary size, special tokens, and an initial byte alphabet. citeturn638725search0turn638725search1

### Prepare E004

Install the new tokenizer dependency:

```powershell
python -m pip install -e ".[dev,data]"
```

Download Dolly only when needed:

```powershell
python scripts/fetch_dolly.py
```

Train the tokenizer and build disk-backed uint16 corpora:

```powershell
python scripts/prepare_e004.py \
  --story-train data/processed/tinystories/train.txt \
  --story-val data/processed/tinystories/val.txt \
  --dolly-source data/raw/dolly/databricks-dolly-15k.jsonl \
  --vocab-size 2048 \
  --tokenizer-story-examples 100000 \
  --out-dir data/processed/e004
```

The Dolly source is the official Databricks dataset, which currently lists 15,015 rows and CC BY-SA 3.0 licensing. citeturn768411search1turn768411search2

### E004 smoke run

```powershell
python scripts/train_e004.py \
  --story-train data/processed/e004/story_train.u16 \
  --story-val data/processed/e004/story_val.u16 \
  --instruction-train data/processed/e004/dolly_train.u16 \
  --instruction-val data/processed/e004/dolly_val.u16 \
  --steps 50 \
  --batch-size 8 \
  --lr 3e-4 \
  --story-weight 0.75 \
  --instruction-weight 0.25 \
  --out checkpoints/e004_smoke.pt \
  --best-out checkpoints/e004_smoke_best.pt
```

Evaluate:

```powershell
python scripts/evaluate_e004.py \
  --checkpoint checkpoints/e004_smoke_best.pt \
  --max-new-tokens 60 \
  --temperature 0.7 \
  --top-k 20
```

The first E004 run is deliberately a **50-step smoke experiment**. No long CPU training is justified until the tokenizer, loss curves, and generations are verified.
