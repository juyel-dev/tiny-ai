"""Evaluate a Banglish -> Bangla checkpoint honestly: not "does the
output look nice", but measured exact-match rate and character-level
accuracy against the held-out validation pairs (scripts/train_banglish.py
never trains on these).

Metrics:
  exact_match   fraction of val examples where generated Bangla ==
                gold Bangla, character-for-character
  char_accuracy 1 - (Levenshtein edit distance / max(len(gold), 1)),
                averaged across examples -- a softer, partial-credit
                measure for when the output is close but not perfect
"""
import argparse
import random

import torch

from tiny_ai.banglish import PROMPT_TEMPLATE, beam_search_generate, levenshtein, load_pairs
from tiny_ai.inference import load_checkpoint


@torch.no_grad()
def generate_bangla(model, tokenizer, prompt_template: str, banglish: str, max_new_tokens: int, temperature: float, top_k: int) -> str:
    device = next(model.parameters()).device
    prompt_ids = [tokenizer.bos_id] + tokenizer.encode(prompt_template.format(banglish=banglish))
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        if top_k:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1) if temperature > 0 else logits.argmax(dim=-1, keepdim=True)
        idx = torch.cat((idx, next_id), dim=1)
        if next_id.item() == tokenizer.eos_id:
            break

    generated = idx[0, len(prompt_ids):].tolist()
    if generated and generated[-1] == tokenizer.eos_id:
        generated = generated[:-1]
    return tokenizer.decode(generated)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pairs-val", required=True)
    p.add_argument("--sample-size", type=int, default=500,
                    help="Evaluate on a random sample of this many val pairs (0 = all).")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy decoding")
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--show-examples", type=int, default=10)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--decoding", choices=["greedy", "beam"], default="greedy")
    p.add_argument("--beam-width", type=int, default=5)
    p.add_argument("--length-penalty", type=float, default=1.0,
                    help="Beam search only. >1 favors longer completions, <1 favors shorter.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_checkpoint(args.checkpoint, device=device)

    # Read the prompt format from the checkpoint itself rather than only
    # the constant imported above -- guarantees eval matches exactly how
    # this particular checkpoint was trained, even across format changes.
    raw_payload = torch.load(args.checkpoint, map_location=device)
    prompt_template = raw_payload.get("prompt_template", PROMPT_TEMPLATE)

    val_pairs = load_pairs(args.pairs_val)
    if args.sample_size and args.sample_size < len(val_pairs):
        random.seed(args.seed)
        val_pairs = random.sample(val_pairs, args.sample_size)

    exact = 0
    char_acc_sum = 0.0
    shown = 0
    for banglish, gold in val_pairs:
        if args.decoding == "beam":
            pred = beam_search_generate(model, tokenizer, prompt_template, banglish,
                                         args.max_new_tokens, args.beam_width, args.length_penalty)
        else:
            pred = generate_bangla(model, tokenizer, prompt_template, banglish, args.max_new_tokens, args.temperature, args.top_k)
        is_exact = pred == gold
        exact += is_exact
        char_acc_sum += 1 - levenshtein(pred, gold) / max(len(gold), 1)

        if shown < args.show_examples:
            mark = "OK " if is_exact else "X  "
            print(f"[{mark}] BN: {banglish}")
            print(f"      gold: {gold}")
            print(f"      pred: {pred}")
            shown += 1

    n = len(val_pairs)
    print()
    print(f"evaluated: {n:,} pairs")
    print(f"exact_match:   {exact / n:.1%}")
    print(f"char_accuracy: {char_acc_sum / n:.1%}")


if __name__ == "__main__":
    main()
