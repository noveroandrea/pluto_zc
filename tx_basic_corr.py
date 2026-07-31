"""
TX: transmit a repeating ZC preamble from a PlutoSDR.
"""

import numpy as np
import adi
import time

URI      = "ip:192.168.2.1"     # default Pluto USB-ethernet address
FS       = int(1e6)             # sample rate (Hz); AD936x minimum is ~521 kHz
LO       = int(2.4e9)           # carrier (Hz) 2.4 GHz ISM band; AD936x max is 6 GHz
BW       = int(1e6)             # analog TX filter bandwidth 1Mhz; AD936x min is ~200 kHz, max is 56 MHz
N        = 1024                 # ZC length
TX_GAIN  = -20                  # dB attenuation, range -89.75 .. 0
SCALE = 2 ** 11                 # 12-bit DAC full scale (~±2048)

def make_zc(n, u=1, q=0):
    k = np.arange(n)
    return np.exp(-1j * np.pi * u * k * (k + 2 * q) / n)


def main():
    sdr = adi.Pluto(URI)

    sdr.sample_rate       = FS
    sdr.tx_lo             = LO
    sdr.tx_rf_bandwidth   = BW
    sdr.tx_hardwaregain_chan0 = TX_GAIN

    # Cyclic buffer: the hardware replays this buffer back-to-back forever,
    # with no host involvement and no gap between repetitions. That is what
    # makes the preamble strictly periodic with period N, which in turn is
    # why the RX side can search each buffer independently.
    sdr.tx_cyclic_buffer = True
    samples = make_zc(N) * SCALE
    sdr.tx(samples)

    print(f"TX: {N}-sample ZC at {LO/1e6:.1f} MHz, fs={FS/1e6:.1f} MS/s, "
          f"gain={TX_GAIN} dB, period {N/FS*1e3:.3f} ms. Ctrl-C to stop.")
    try:
        n = 0
        while True:
            time.sleep(1)
            n += 1
            print(f"  TX running: {n} s, ~{n*FS/N:.0f} preambles sent")
    except KeyboardInterrupt:
        sdr.tx_destroy_buffer()

    # # bad cyclic just for testing
    # sdr.tx_cyclic_buffer = False
    # samples = make_zc(N) * SCALE
    # try:
    #     i = 0
    #     while True:
    #         sdr.tx(samples)
    #         print(f"TX burst {i}")
    #         i += 1
    #         time.sleep(3)
    # except KeyboardInterrupt:
    #     sdr.tx_destroy_buffer()


if __name__ == "__main__":
    main()