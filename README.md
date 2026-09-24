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

## Why does generation look like garbage?

Two common causes, roughly in order of likelihood:

1. **You trained on `data/example.txt`.** That file is a pipeline smoke-test corpus (a few hundred bytes), not training data — see the note in Quick start. No model, tiny or otherwise, learns language from that. Use E002 with a real corpus (TinyStories or your own) instead.
2. **Byte-level tokenization is expensive for a model this small.** The default tokenizer maps text to raw UTF-8 bytes, so a single English word costs 4–8 tokens instead of 1. At E002's ~1.3M parameters, most of that budget goes into re-deriving spelling rather than learning structure. The BPE tokenizer below is the single highest-leverage fix for coherence at a fixed parameter budget — try it before scaling up model size.

Also check: are you actually training long enough? 5,000 steps at batch size 16 is ~20M bytes of exposure — undertrained relative to a multi-hundred-MB corpus. Watch `val_loss` in the training log; if it's still dropping steadily when training stops, run more steps before concluding the architecture is the problem.

## Optional: BPE tokenizer

`tiny_ai/bpe_tokenizer.py` is a small, dependency-free byte-level BPE tokenizer trained directly on your corpus (no network access, no model hub — fully reproducible from `(corpus file, vocab size)`). It keeps the byte tokenizer's lossless fallback for anything unseen, while letting common substrings collapse to a single token.

```bash
# 1. Train a tokenizer on your prepared corpus
python scripts/train_tokenizer.py \
  --corpus data/processed/tinystories/train.txt \
  --vocab-size 4096 \
  --out tokenizer/e002_bpe4096.json

# 2. Encode train/val splits into binary token-id files
python scripts/tokenize_corpus.py \
  --tokenizer tokenizer/e002_bpe4096.json \
  --input data/processed/tinystories/train.txt \
  --output data/processed/tinystories/train.bpe.bin
python scripts/tokenize_corpus.py \
  --tokenizer tokenizer/e002_bpe4096.json \
  --input data/processed/tinystories/val.txt \
  --output data/processed/tinystories/val.bpe.bin

# 3. Train with the BPE tokenizer (also picks up LR warmup + cosine decay, see below)
python scripts/train_e002.py \
  --train data/processed/tinystories/train.bpe.bin \
  --val data/processed/tinystories/val.bpe.bin \
  --tokenizer bpe --bpe-merges tokenizer/e002_bpe4096.json \
  --steps 5000 \
  --out checkpoints/e002_bpe.pt
```

`scripts/evaluate_e002.py` and `scripts/chat.py` read the tokenizer type back out of the checkpoint automatically — nothing to pass at generation time. Checkpoints saved before this existed still load as the original byte tokenizer, unchanged.

Token ids above 255 don't fit in one byte, so BPE-encoded corpora are stored as little-endian uint16 (`tiny_ai.data.MappedTokenIds`) instead of the raw-byte `.txt` files `MappedTokens` reads — that's what `tokenize_corpus.py`'s `.bin` output is for.

## Learning-rate schedule

`scripts/train_e002.py` now uses linear warmup followed by cosine decay (`tiny_ai/schedule.py`) instead of a constant LR, controlled by `--warmup-steps` (default 200) and `--min-lr-ratio` (default 0.1, i.e. decays to 10% of `--lr`). This is a training-loop change only — it applies regardless of which tokenizer you use.

## Training on GitHub Actions

`.github/workflows/train_e002.yml` is a manually-triggered workflow (Actions tab → "train-e002" → Run workflow) that runs the E002 pipeline end-to-end on a GitHub-hosted CPU runner: fetch TinyStories, prepare the corpus, optionally train a BPE tokenizer, train, upload the checkpoint as an artifact.

Worth knowing before you use it:
- **No GPU.** GitHub-hosted runners are CPU-only. Fine for E002's current scale (~1.3M params), not something you'd want for a much bigger model.
- **Free, but capped.** Standard Linux runners are unlimited-minutes on public repos, but a single job is capped at 6 hours; the workflow sets `timeout-minutes: 340` to leave a safety margin.
- **Progress survives a timeout.** `--save-every` (default 250 steps) writes a resumable checkpoint — weights, optimizer state, and step count — and the artifact-upload step runs with `if: always()`, so a killed job still yields something to resume from. Re-run the workflow with `resume_from_run_id` set to the earlier run's ID to continue.
- **BPE tokenizer training is pure Python** (see above) and its cost scales with the number of distinct words in the corpus, not corpus size — but TinyStories' full training split is large enough that this can still take a while. Use the `bpe_max_chars` input to cap it, or time `scripts/train_tokenizer.py` locally on a subset first before committing to a full run.

The resulting checkpoint is only downloadable from the workflow's artifact, not automatically committed to the repo — download it via the Actions run page and check it into `checkpoints/` (or wherever you keep them) yourself if you want it in version control.

## Experiment: Banglish -> Bangla transliteration

A second, separate experiment (E002's story model is untouched) built on the same infra: `tiny_ai/banglish.py`, `scripts/prepare_banglish_data.py`, `scripts/train_banglish.py`, `scripts/evaluate_banglish.py`, and `.github/workflows/train_banglish.yml` train a small model to convert romanized Bangla ("ami tomake bhalobashi") into Bangla script ("আমি তোমাকে ভালোবাসি"), using [BanglaTLit](https://huggingface.co/datasets/aplycaebous/BanglaTLit) (Fahim et al., EMNLP'24 Findings).

Two phases: plain language-model pretraining on romanized Bangla broadly, then fine-tuning on ~40K labeled pairs with completion-only loss (the model is only trained to predict the Bangla output, not the Banglish prompt it's given — see `tiny_ai/banglish.py`'s packing/masking). Evaluated honestly on held-out pairs: exact-match rate and Levenshtein-based character accuracy, not vibes.

### Live demo

`scripts/export_banglish_web.py` + `scripts/build_banglish_web.py` package a trained checkpoint into one self-contained HTML file that runs entirely in the browser — the forward pass is hand-implemented in JavaScript (mirroring `tiny_ai/model.py` exactly, cross-checked byte-for-byte against the Python model's output on real held-out pairs before shipping), with a KV-cache for interactive latency. No server, no API calls, works offline once loaded. Deployed via `.github/workflows/deploy_banglish_web.yml` to GitHub Pages.

To build it yourself from a checkpoint:
```bash
python scripts/export_banglish_web.py --checkpoint checkpoints/banglish.pt --out-dir web_export
python scripts/build_banglish_web.py --export-dir web_export --out web/banglish.html
```
The weights are embedded as base64-encoded float16 (halves the size vs float32, negligible quality loss at this model scale) — keep the exported model small enough to stay under the ~16 MiB a single HTML file can reasonably hold.
