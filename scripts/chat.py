import argparse
from tiny_ai.inference import load_checkpoint, generate

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--prompt", default=None)
args = p.parse_args()

model, tokenizer = load_checkpoint(args.checkpoint)

if args.prompt is not None:
    print(generate(model, tokenizer, args.prompt))
else:
    print("Tiny-AI E001. Type 'exit' to quit.")
    while True:
        prompt = input("You: ")
        if prompt.strip().lower() == "exit":
            break
        print("Tiny-AI:", generate(model, tokenizer, prompt))
