"""Assemble tiny_ai/static/banglish_template.html with a real
manifest.json + weights.bin (from scripts/export_banglish_web.py) into
one self-contained HTML file -- no build step needed at serve time,
just open the output file (or publish it) directly.
"""
import argparse
import base64
import json
from pathlib import Path

TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "tiny_ai" / "static" / "banglish_template.html"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export-dir", default="web/banglish", help="Directory from export_banglish_web.py")
    p.add_argument("--out", default="web/banglish.html")
    args = p.parse_args()

    export_dir = Path(args.export_dir)
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    weights_bytes = (export_dir / "weights.bin").read_bytes()
    weights_b64 = base64.b64encode(weights_bytes).decode("ascii")

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace("__MANIFEST_JSON__", json.dumps(manifest)).replace("__WEIGHTS_B64__", weights_b64)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    size_mb = out.stat().st_size / 1024**2
    print(f"wrote: {out} ({size_mb:.2f} MiB)")
    if size_mb > 16:
        print("WARNING: exceeds the 16 MiB single-file artifact cap.")


if __name__ == "__main__":
    main()
