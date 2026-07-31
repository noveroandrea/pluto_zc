"""
RX: capture from a PlutoSDR and report ZC preamble timing, CFO, match quality.
"""

import numpy as np
import adi

URI        = "ip:192.168.3.1"
FS         = int(1e6)
LO       = int(2.4e9)           # carrier (Hz) 2.4 GHz ISM band; AD936x max is 6 GHz
BW       = int(1e6)             # analog TX filter bandwidth 1Mhz; AD936x min is ~200 kHz, max is 56 MHz
N          = 1024
L          = 32                  # coherent blocks; must divide N, due to freq difference, if perfectly coherent => i would not need, 
                                # the condition is that phase rotation across each block stays under 0.25 cycles => df * M / fs < 0.25, where M = N/L is the number of samples per block
PSR_THRESH = 6.0                 #peak-to-sidelobe ratio threshold for declaring a preamble found
BUF        = 4 * N              # >= 2N-1 so a periodic ZC is always whole


def make_zc(n, u=1, q=0):
    k = np.arange(n)
    return np.exp(-1j * np.pi * u * k * (k + 2 * q) / n)


class ZCDetector:
    """Segmented matched filter: coherent within N/L-sample blocks, magnitudes
    summed across blocks. Tolerates ~L x more CFO than a full coherent
    correlation, at a cost of ~sqrt(L) in processing gain."""

    def __init__(self, ref, L=8, psr_thresh=8.0):
        self.ref, self.N, self.L = np.asarray(ref, complex), len(ref), L
        if self.N % L:
            raise ValueError(f"L={L} must divide N={self.N}")
        self.M = self.N // L
        self.psr_thresh = psr_thresh
        self.blocks = [self.ref[l*self.M:(l+1)*self.M] for l in range(L)]

    def detect(self, y):
        y = np.asarray(y, complex)
        K = len(y) - self.N + 1
        if K < 1:
            return dict(found=False, lag=None, psr=0.0, cfo=0.0)

        c = np.empty((self.L, K), complex)
        for l, blk in enumerate(self.blocks):
            full = np.correlate(y, blk, mode="valid")
            c[l] = full[l*self.M : l*self.M + K]     # realign to preamble start

        score = np.abs(c).sum(axis=0)                # non-coherent combining
        k     = int(np.argmax(score))
        psr   = float(score[k] / (np.median(score) + 1e-12))

        cfo = 0.0
        if self.L > 1:
            prod = np.sum(c[1:, k] * np.conj(c[:-1, k]))
            cfo  = float(np.angle(prod) / (2*np.pi*self.M))   # cycles/sample
        return dict(found=psr > self.psr_thresh, lag=k, psr=psr, cfo=cfo)

    def refine(self, y, lag, cfo):
        seg = y[lag:lag+self.N] * np.exp(-2j*np.pi*cfo*np.arange(self.N))
        return float(np.abs(np.vdot(self.ref, seg)) /
                     (np.linalg.norm(self.ref)*np.linalg.norm(seg) + 1e-12))


def main():
    sdr = adi.Pluto(URI)

    sdr.sample_rate       = FS
    sdr.rx_lo             = LO
    sdr.rx_rf_bandwidth   = BW
    sdr.rx_buffer_size    = BUF

    # Manual gain keeps the scale constant across buffers, which matters if you
    # ever compare raw magnitudes. Switch to "slow_attack" for over-the-air work
    # where the level varies; the detector's metrics are gain-invariant anyway.
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0   = 40        # dB, range 0 .. 71

    sdr.rx_destroy_buffer()                 # clear any stale buffer config

    det = ZCDetector(make_zc(N), L=L, psr_thresh=PSR_THRESH)
    M   = N // L # samples per block, the // does integer division so the unambiguous CFO range is df * M / fs << 1

    print(f"RX: fs={FS/1e6:.1f} MS/s, unambiguous CFO range "
          f"+/-{FS/(2*M)/1e3:.1f} kHz")

    while True:
        y = sdr.rx()                        # complex64, |.| up to ~2048 (12-bit)
        r = det.detect(y)
        if r["found"]:
            q = det.refine(y, r["lag"], r["cfo"])
            print(f"ZC @ lag {r['lag']:5d} | PSR {r['psr']:6.1f} | "
                  f"CFO {r['cfo']*FS:+9.1f} Hz | match {q:.4f}")
        else:
            print(f"no preamble (PSR {r['psr']:.1f})")
if __name__ == "__main__":
    main()