"""
TX: push a ZC preamble to the emulated Pluto TX device (writes into data.bin).

Run the emulator first, then:  python tx.py
"""

import iio
import numpy as np
import time
from time import sleep

URI = "ip:192.168.2.1"
TX_DEV = "cf-ad9361-dds-core-lpc"   # iio:device2
N = 1024
SCALE = 2 ** 14


def make_zc(n, u=1, q=0):
    k = np.arange(n)
    return np.exp(-1j * np.pi * u * k * (k + 2 * q) / n)


def to_int16_iq(x):
    xi = np.round(np.real(x) * SCALE).astype(np.int16)
    xq = np.round(np.imag(x) * SCALE).astype(np.int16)
    iq = np.empty(2 * len(x), dtype=np.int16)
    iq[0::2] = xi   # I -> voltage0
    iq[1::2] = xq   # Q -> voltage1
    return iq


def main():
    ctx = iio.Context(URI)
    tx = ctx.find_device(TX_DEV)
    assert tx is not None, f"missing {TX_DEV}"


    ch_i = tx.find_channel("voltage0", True)   # True = output
    ch_q = tx.find_channel("voltage1", True)
    ch_i.enabled = True
    ch_q.enabled = True

    sent = to_int16_iq(make_zc(N))
    buf = iio.Buffer(tx, N, False)             # cyclic=False
    while(1):
        buf.write(bytearray(sent.tobytes()))
        buf.push()
        print(f"TX: pushed {N} ZC samples ({sent.nbytes} bytes) into data.bin")
        # Sleep for a bit to avoid overwhelming the buffer
        time.sleep(3)


if __name__ == "__main__":
    main()
