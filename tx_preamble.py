"""
TX: transmit a repeating LoRa-style synchronisation preamble from a PlutoSDR.

Preamble (10 slots of N samples, replayed cyclically):
    4 x up-chirp | 2 x up-chirp modulated with S_SYNC | 4 x down-chirp

The waveform functions below must stay byte-identical to those in rx_basic.py,
otherwise the dechirping produces no clean tone. Keeping them duplicated is
deliberate for now so each script runs standalone; move them to a shared module
once the parameters stop changing.
"""

import numpy as np
import adi
import time

URI      = "ip:192.168.2.1"     # default Pluto USB-ethernet address
LO       = int(2.4e9)           # carrier (Hz) 2.4 GHz ISM band; AD936x max is 6 GHz
FS       = int(10e6)             # sample rate (Hz); AD936x minimum is ~521 kHz
BW=FS*1.5
TX_GAIN  = -50                  # dB attenuation, range -89.75 .. 0
SCALE    = 2 ** 14              # 12-bit DAC full scale (~±2048)

N        = 2**10                # samples per chirp (minimum sampling rate)
S_SYNC   = 200                   # sync-word symbol value
N_LEN     = 6                    # unmodulated up-chirps and downchirps
N_UP     = N_LEN                # unmodulated up-chirps
N_SYNC   = 2                    # sync-word chirps
N_DOWN   = N_LEN                # down-chirps
# NOTE: with arbitrary window alignment a region of R chirps yields only R-1
# fully contained dechirp slots, so N_UP = 4 leaves exactly the 3 slots the
# receiver requires and no margin for a noisy slot. Raising N_UP and N_DOWN to
# 6 is advisable once the link works -- but they must be changed in rx_basic.py
# at the same time, since the receiver derives the region lengths from them.


def make_chirp(n, up=True):
    """
    Base chirp, B[k] = exp(j2*pi*(k^2/2N - k/2)).

    Constant envelope, so every sample sits at SCALE and there is no peak to
    back off for. The down-chirp is the complex conjugate.
    """
    k = np.arange(n)
    c = np.exp(1j * 2 * np.pi * (k**2 / (2 * n) - k / 2))
    return c if up else np.conj(c)


def make_symbol(n, s, up=True):
    """
    Chirp modulated with symbol s, i.e. a cyclic shift of the base chirp by s.

    At the receiver this dechirps to a tone at normalised frequency s/N. The
    sync word exists so that this known shift can be checked against the
    measured bin separation, validating detection independently of the STO and
    CFO estimates (both of which cancel in the difference).
    """
    return np.roll(make_chirp(n, up), -int(s))


def make_preamble(n=N, s=S_SYNC):
    """Full 10N preamble. Length is a multiple of N, which matters: it keeps the
    receiver's slot grid at the same offset for every cyclic repetition."""
    up, dn = make_chirp(n, True), make_chirp(n, False)
    sync = make_symbol(n, s, True)
    return np.concatenate([np.tile(up,   N_UP),
                           np.tile(sync, N_SYNC),
                           np.tile(dn,   N_DOWN)])


def main():
    sdr = adi.Pluto(URI)

    sdr.sample_rate       = FS
    sdr.tx_lo             = LO
    sdr.tx_rf_bandwidth   = BW
    sdr.tx_hardwaregain_chan0 = TX_GAIN

    # Cyclic buffer: the hardware replays this buffer back to back forever, with
    # no host involvement and no gap between repetitions. The transmitted signal
    # is therefore strictly periodic with period 10N, which is what allows the
    # receiver to search each captured buffer independently.
    sdr.tx_cyclic_buffer = True

    samples = make_preamble() * SCALE
    #add a sequence of data each data is 10 blank samples and a single upchirp, repeat this sequence 10 times

    # data_sequence = np.concatenate( [np.zeros(30)],(make_chirp(N, True),make_chirp(N,True),make_chirp(N,True),[np.zeros(10)]) * 10)
    # samples = np.concatenate([samples, data_sequence])
    sdr.tx(samples)

    n_chirps = N_UP + N_SYNC + N_DOWN
    print(f"TX: {n_chirps}-chirp preamble ({len(samples)} samples) at "
          f"{LO/1e9:.3f} GHz, fs={FS/1e6:.1f} MS/s, gain={TX_GAIN} dB, "
          f"period {len(samples)/FS*1e3:.3f} ms. Ctrl-C to stop.")
    try:
        t = 0
        while True:
            time.sleep(1)
            t += 1
            print(f"  TX running: {t} s, ~{t*FS/len(samples):.0f} preambles sent")
    except KeyboardInterrupt:
        sdr.tx_destroy_buffer()


if __name__ == "__main__":
    main()