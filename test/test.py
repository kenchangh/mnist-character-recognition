# SPDX-FileCopyrightText: © 2026 kenchangh
# SPDX-License-Identifier: Apache-2.0
"""Cocotb tests for the MNIST classifier.

Feeds real MNIST test images (exported by train/train.py together with the
integer reference model's predictions) through the serial protocol and
checks the RTL digit is bit-exact against the reference model.

Select the variant with `make VARIANT=linear|mlp16|mlp32` (default mlp32);
the matching test/vectors_<variant>.json is used as ground truth.
"""

import json
import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge

VARIANT = os.environ.get("VARIANT", "mlp32")
# Latency after the 64th bit: 64 (L1) + 1 (BIN) + H (L2) + 10 (argmax) + slack
DONE_TIMEOUT = 250

DATA, SHIFT_EN, START = 1 << 0, 1 << 1, 1 << 2


def load_vectors():
    path = os.path.join(os.path.dirname(__file__), f"vectors_{VARIANT}.json")
    with open(path) as f:
        return json.load(f)


async def setup(dut):
    clock = Clock(dut.clk, 100, unit="ns")  # 10 MHz
    cocotb.start_soon(clock.start())
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


async def drive(dut, value, cycles=1):
    """Change inputs away from the sampling (rising) edge."""
    await FallingEdge(dut.clk)
    dut.ui_in.value = value
    for _ in range(cycles - 1):
        await FallingEdge(dut.clk)


async def send_start(dut):
    await drive(dut, START)
    await drive(dut, 0)


async def send_bits(dut, word, nbits=64):
    """Send bit 0 (pixel 0, top-left) first."""
    for i in range(nbits):
        bit = (word >> i) & 1
        await drive(dut, SHIFT_EN | (DATA if bit else 0))
    await drive(dut, 0)


async def wait_done(dut):
    for _ in range(DONE_TIMEOUT):
        await ClockCycles(dut.clk, 1)
        if (int(dut.uo_out.value) >> 7) & 1:
            return int(dut.uio_out.value) & 0xF
    raise AssertionError("DONE did not assert within timeout")


async def classify(dut, word):
    await send_start(dut)
    await send_bits(dut, word)
    return await wait_done(dut)


@cocotb.test()
async def test_accuracy(dut):
    """Feed all exported MNIST test images; require bit-exact agreement with
    the integer reference model and report accuracy vs true labels."""
    vec = load_vectors()
    await setup(dut)

    correct = 0
    for n, (img_hex, expected, label) in enumerate(
            zip(vec["images"], vec["expected"], vec["labels"])):
        digit = await classify(dut, int(img_hex, 16))
        assert digit == expected, (
            f"image {n}: RTL={digit} reference={expected} (label={label})")
        correct += digit == label
        # done flag must clear on the next START (checked implicitly each loop)

    n_total = len(vec["images"])
    acc = correct / n_total
    dut._log.info(
        f"[{VARIANT}] RTL bit-exact on {n_total}/{n_total} images; "
        f"MNIST accuracy {correct}/{n_total} = {acc:.1%}")


@cocotb.test()
async def test_reset_midload(dut):
    """rst_n asserted mid-load must cleanly restart: a full reload after
    reset produces the correct result."""
    vec = load_vectors()
    word = int(vec["images"][0], 16)
    await setup(dut)

    # partially load garbage, then hard reset
    await send_start(dut)
    await send_bits(dut, 0xFFFF_FFFF_FFFF_FFFF, nbits=30)
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    assert (int(dut.uo_out.value) >> 7) & 1 == 0, "done must clear on reset"
    assert int(dut.uo_out.value) & 0x7F == 0, "7-seg must blank on reset"

    digit = await classify(dut, word)
    assert digit == vec["expected"][0], "classification wrong after mid-load reset"


@cocotb.test()
async def test_start_midload(dut):
    """A new START mid-load must restart the bit counter (soft restart)."""
    vec = load_vectors()
    word = int(vec["images"][1], 16)
    await setup(dut)

    await send_start(dut)
    await send_bits(dut, 0xAAAA_AAAA_AAAA_AAAA, nbits=17)  # abandoned load
    digit = await classify(dut, word)
    assert digit == vec["expected"][1], "classification wrong after START mid-load"


@cocotb.test()
async def test_back_to_back(dut):
    """Several classifications in a row without reset; done/busy behave."""
    vec = load_vectors()
    await setup(dut)

    for n in range(5):
        word = int(vec["images"][n], 16)
        await send_start(dut)
        assert (int(dut.uio_out.value) >> 5) & 1 == 0, "busy must be low in LOAD"
        await send_bits(dut, word)
        digit = await wait_done(dut)
        assert digit == vec["expected"][n]
        assert (int(dut.uio_out.value) >> 4) & 1 == 1, "uio done mirrors uo done"
        # result must hold until the next START
        await ClockCycles(dut.clk, 20)
        assert (int(dut.uio_out.value) & 0xF) == vec["expected"][n]


@cocotb.test()
async def test_shift_en_gating(dut):
    """Clock cycles with SHIFT_EN low must not shift data in."""
    vec = load_vectors()
    word = int(vec["images"][2], 16)
    await setup(dut)

    await send_start(dut)
    # send with gaps: every bit followed by idle cycles
    for i in range(64):
        bit = (word >> i) & 1
        await drive(dut, SHIFT_EN | (DATA if bit else 0))
        await drive(dut, DATA, cycles=2)  # data present but not enabled
    await drive(dut, 0)
    digit = await wait_done(dut)
    assert digit == vec["expected"][2], "SHIFT_EN gating broken"
