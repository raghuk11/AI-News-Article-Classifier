"""Classify headlines from the command line.

    python predict.py "Stocks rally as central bank holds rates"
    python predict.py --file sample_headlines.txt
    python predict.py                      # uses the sample headlines
"""
import argparse
from pathlib import Path

import numpy as np

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="*", help="one or more headlines (quote each)")
    ap.add_argument("--file", help="text file with one headline per line")
    args = ap.parse_args()

    if not C.artifacts_exist():
        raise SystemExit("No trained model found. Run:  python train.py --quick")

    if args.file:
        texts = [l.strip() for l in Path(args.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    elif args.texts:
        texts = args.texts
    else:
        texts = [l.strip() for l in Path(__file__).with_name("sample_headlines.txt")
                 .read_text(encoding="utf-8").splitlines() if l.strip()]

    model, vec = C.load_artifacts()
    probs = C.predict_texts(model, vec, texts)
    for text, p in zip(texts, probs):
        k = int(np.argmax(p))
        print(f"{C.CLASS_NAMES[k]:9} ({p[k]:.2f}) | {text}")


if __name__ == "__main__":
    main()
