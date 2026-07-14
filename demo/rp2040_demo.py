"""MicroPython demo for the Tiny Tapeout demo board (RP2040).

Sends an 8x8 binarized MNIST image to tt_um_kenchangh_mnist over the
serial pixel protocol and reads back the predicted digit. The digit also
appears on the demo board's 7-segment display (decimal point = DONE).

Copy this file to the demo board (e.g. with mpremote) and run:

    import rp2040_demo
    rp2040_demo.run()

Requires the Tiny Tapeout MicroPython SDK (ttboard) that ships on the
demo board's RP2040.

Pin map (chip side):
  ui[0] DATA, ui[1] SHIFT_EN, ui[2] START
  uio[3:0] digit, uio[4] DONE, uio[5] BUSY  (all driven by the chip)
"""

from ttboard.demoboard import DemoBoard

DATA, SHIFT_EN, START = 1 << 0, 1 << 1, 1 << 2
BIN_THRESHOLD = 0.15  # must match training (train/preprocess.py)

# A few real MNIST test images (64-bit hex, pixel 0 = top-left, bit i =
# row-major pixel i) with the label the chip should output.
SAMPLES = [
    (0x00003C3C2C081810, 6),
    (0x000C3C3820203800, 2),
    (0x003C64303C303C00, 3),
    (0x00047E3C20203818, 2),
]


def image_to_word(img28):
    """Preprocess a 28x28 image (list of lists, values 0..1) into the 64-bit
    input word: adaptive 8x8 average pool, binarize, pack row-major."""
    word = 0
    for r in range(8):
        r0, r1 = (r * 28) // 8, -((-(r + 1) * 28) // 8)
        for c in range(8):
            c0, c1 = (c * 28) // 8, -((-(c + 1) * 28) // 8)
            acc = n = 0
            for y in range(r0, r1):
                for x in range(c0, c1):
                    acc += img28[y][x]
                    n += 1
            if acc / n > BIN_THRESHOLD:
                word |= 1 << (r * 8 + c)
    return word


def show(word):
    """Print the 8x8 image as ASCII art."""
    for r in range(8):
        print("".join("#" if (word >> (r * 8 + c)) & 1 else "." for c in range(8)))


def classify(tt, word):
    """Send one image, return the predicted digit."""
    # Manual clocking so every input change meets setup before the edge.
    tt.clock_project_stop()

    tt.ui_in.value = START          # START strobe
    tt.clock_project_once()
    tt.ui_in.value = 0
    tt.clock_project_once()

    for i in range(64):             # 64 pixel bits, LSB (pixel 0) first
        bit = (word >> i) & 1
        tt.ui_in.value = SHIFT_EN | (DATA if bit else 0)
        tt.clock_project_once()
    tt.ui_in.value = 0

    for _ in range(300):            # compute takes ~172 cycles
        tt.clock_project_once()
        if tt.uio_in.value & (1 << 4):     # DONE
            return tt.uio_in.value & 0xF
    raise RuntimeError("DONE never asserted — is the right project selected?")


def run():
    tt = DemoBoard.get()
    tt.shuttle.tt_um_kenchangh_mnist.enable()
    tt.uio_oe_pico.value = 0        # chip drives uio[5:0]; RP2040 must not
    tt.reset_project(True)
    tt.clock_project_once()
    tt.reset_project(False)

    for word, label in SAMPLES:
        show(word)
        digit = classify(tt, word)
        print("chip says: %d (expected %d) %s\n"
              % (digit, label, "OK" if digit == label else "MISMATCH"))

    # Leave the clock running so the 7-seg display keeps showing the result.
    tt.clock_project_PWM(1_000_000)


if __name__ == "__main__":
    run()
