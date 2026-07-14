![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg) ![](../../workflows/fpga/badge.svg)

# MNIST Digit Recognition on Tiny Tapeout (IHP 130nm)

Self-contained MNIST handwritten-digit classifier with **all weights on-chip**,
baked into the netlist as ternary ({-1, 0, +1}) Verilog constants and
constant-folded at synthesis — no RAM, no ROM, no weight loading.

- [Project datasheet (protocol + timing diagram)](docs/info.md)
- 8×8 binarized input, shifted in serially (64 bits), digit out on a
  7-segment display + binary on `uio[3:0]`
- Bit-serial ternary MAC engine: ~24 µs per classification at 10 MHz

| Variant (synthesis parameter) | MNIST accuracy* | Cell area (yosys, sg13g2) |
|---|---|---|
| **MLP 64→32→10** (`MLP=1, HIDDEN=32`, default) | **80.3 %** | ~34k µm² (~47 % of a 2×2 tile) |
| linear 64→10 (`MLP=0`) | 78.3 % | ~20k µm² |
| MLP 64→16→10 (`MLP=1, HIDDEN=16`) | 76.0 % | ~29k µm² |

\* exact integer model, full 10,000-image test set. The RTL is verified
bit-exact against that model on 200 images per variant (cocotb).

## Repo layout

| Path | Purpose |
|---|---|
| `train/train.py` | PyTorch quantization-aware training (ternary weights, binary activations), exports JSON weights + test vectors |
| `train/gen_weights.py` | JSON weights → `src/weights.v` (constants) |
| `train/preprocess.py` | The single source of truth for 28×28 → 64-bit preprocessing |
| `src/project.v` | FSM + bit-serial compute core (variant parameters `MLP`, `HIDDEN`) |
| `src/weights.v` | Generated weight constants (all three variants) |
| `test/` | cocotb testbench: 200 real MNIST images bit-exact, reset/restart/gating tests |
| `demo/rp2040_demo.py` | MicroPython demo for the TT demo board |

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install torch torchvision numpy cocotb pytest
.venv/bin/python train/train.py          # trains all variants, ~5 min CPU
.venv/bin/python train/gen_weights.py    # regenerates src/weights.v
cd test && make -B VARIANT=mlp32         # also: VARIANT=mlp16, VARIANT=linear
```

The shipped variant is chosen by the parameter defaults in `src/project.v`
(`MLP=1, HIDDEN=32`); keep `VARIANT` in `test/Makefile` in sync so the
gate-level CI test checks the same variant.

## What is Tiny Tapeout?

Tiny Tapeout is an educational project that aims to make it easier and cheaper than ever to get your digital and analog designs manufactured on a real chip.

To learn more and get started, visit https://tinytapeout.com.

The GitHub action will automatically build the ASIC files using [LibreLane](https://www.zerotoasiccourse.com/terminology/librelane/).

## Resources

- [FAQ](https://tinytapeout.com/faq/)
- [Digital design lessons](https://tinytapeout.com/digital_design/)
- [Build your design locally](https://www.tinytapeout.com/guides/local-hardening/)
- [Join the community](https://tinytapeout.com/discord)

## What next?

- [Submit your design to the next shuttle](https://app.tinytapeout.com/).
