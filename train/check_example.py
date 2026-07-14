"""Sanity-check the exported weights on real MNIST images, step by step.

Usage:
    .venv/bin/python train/check_example.py [test-set index ...]

For each image this shows: the 28x28 -> 8x8 binarized input, the 64-bit
word exactly as it would be shifted into the chip, the ten class
accumulator values the RTL computes, and the predicted vs true digit.
It also cross-checks the prediction against test/vectors_*.json (the
ground truth the cocotb testbench verified the RTL against, bit-exact).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import BIN_THRESHOLD, binarize, pack_bits, pool_8x8

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def int_scores(wts, bits):
    """Exact twin of the RTL datapath, returning all intermediate values."""
    acc1 = np.array(wts["b1"]) + bits @ np.array(wts["w1"]).T
    if "w2" in wts:
        h = (acc1 >= 0).astype(int)
        scores = np.array(wts["b2"]) + h @ np.array(wts["w2"]).T
        return scores, h
    return acc1, None


def main():
    from torchvision import datasets
    test = datasets.MNIST(os.path.join(ROOT, "data"), train=False, download=True)
    images = test.data.numpy().astype(np.float64) / 255.0
    labels = test.targets.numpy()

    variant = os.environ.get("VARIANT", "mlp32")
    with open(os.path.join(ROOT, "weights", f"{variant}.json")) as f:
        wts = json.load(f)
    with open(os.path.join(ROOT, "test", f"vectors_{variant}.json")) as f:
        vec = json.load(f)
    word_to_expected = {int(w, 16): e for w, e in zip(vec["images"], vec["expected"])}

    indices = [int(a) for a in sys.argv[1:]] or [0, 1, 2, 3]
    for idx in indices:
        img8 = binarize(pool_8x8(images[idx]), BIN_THRESHOLD)
        word = pack_bits(img8)
        scores, h = int_scores(wts, img8.reshape(64))
        pred = int(np.argmax(scores))

        print(f"=== MNIST test image #{idx} (true label: {labels[idx]}) ===")
        for row in img8:
            print("   " + "".join("#" if p else "." for p in row))
        print(f"   64-bit input word: 0x{word:016x}  (send bit 0 first)")
        if h is not None:
            print(f"   hidden bits: {''.join(str(b) for b in h)}")
        print("   class scores:", "  ".join(f"{d}:{s:+d}" for d, s in enumerate(scores)))
        print(f"   predicted: {pred}  ->  "
              + ("CORRECT" if pred == labels[idx] else f"WRONG (true {labels[idx]})"))
        if word in word_to_expected:
            ok = word_to_expected[word] == pred
            print(f"   cross-check vs test/vectors_{variant}.json (RTL-verified): "
                  + ("match" if ok else "MISMATCH"))
        print()


if __name__ == "__main__":
    main()
