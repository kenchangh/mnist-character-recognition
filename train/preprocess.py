"""Shared MNIST preprocessing: 28x28 grayscale -> 64-bit binary vector.

This is the single source of truth for how images are prepared, used by:
  - train/train.py           (training + weight/vector export)
  - test/test.py             (cocotb RTL verification)
  - demo/rp2040_demo.py      (re-implemented in pure Python for MicroPython)

Pipeline:
  1. Input: 28x28 array of floats in [0, 1] (MNIST pixel / 255).
  2. Adaptive average pool to 8x8: bin r covers rows floor(r*28/8) .. ceil((r+1)*28/8)-1,
     same for columns (this matches torch.nn.functional.adaptive_avg_pool2d).
  3. Binarize: pixel = 1 if bin average > BIN_THRESHOLD else 0.
  4. Pack row-major into a 64-bit integer: bit (row*8 + col), row 0 = top,
     col 0 = left. Bit 0 is sent to the chip FIRST (LSB-first serial load).
"""

import numpy as np

BIN_THRESHOLD = 0.15  # overwritten by train.py sweep result; keep in sync


def pool_8x8(img28: np.ndarray) -> np.ndarray:
    """Adaptive average pool a 28x28 float array to 8x8."""
    out = np.zeros((8, 8), dtype=np.float64)
    for r in range(8):
        r0, r1 = (r * 28) // 8, -((-(r + 1) * 28) // 8)  # floor, ceil
        for c in range(8):
            c0, c1 = (c * 28) // 8, -((-(c + 1) * 28) // 8)
            out[r, c] = img28[r0:r1, c0:c1].mean()
    return out


def binarize(img8: np.ndarray, threshold: float = BIN_THRESHOLD) -> np.ndarray:
    return (img8 > threshold).astype(np.uint8)


def pack_bits(bits8x8: np.ndarray) -> int:
    """Pack an 8x8 binary array into a 64-bit int, bit index = row*8 + col."""
    v = 0
    flat = bits8x8.reshape(64)
    for i in range(64):
        if flat[i]:
            v |= 1 << i
    return v


def image_to_word(img28: np.ndarray, threshold: float = BIN_THRESHOLD) -> int:
    return pack_bits(binarize(pool_8x8(img28), threshold))
