"""
RX: read samples from the Pluto RX device and detect the ZC preamble.
"""

import iio
import numpy as np


URI = "ip:192.168.3.1"
RX_DEV = "cf-ad9361-lpc"            # iio:device3
N = 1024
L = 8                               # coherent blocks; N must be divisible by L
PSR_THRESH = 8.0
BUF_SAMPLES = 4 * N                 # >= 2N so a periodic preamble is always whole
SCALE = 2 ** 14


def make_zc(n, u=1, q=0):
    """
    Generate a Zadoff-Chu sequence of length n, root u, shift q.

    x[k] = exp(-j*pi*u*k*(k+2q)/n) -- unit-magnitude complex baseband IQ,
    used directly as the transmitted preamble and as the correlator
    reference here. Must match tx.py exactly (same n, u, q) or the
    matched filter produces no peak.
    """
    k = np.arange(n)
    return np.exp(-1j * np.pi * u * k * (k + 2 * q) / n)


def from_int16_iq(raw):
    """
    Unpack a raw libiio buffer into complex baseband samples.

    The device delivers interleaved 16-bit I,Q pairs (I0,Q0,I1,Q1,...).
    De-interleave, combine into complex, and rescale by SCALE to undo the
    fixed-point scaling tx.py applied, so the result is back in the same
    units as make_zc() (roughly unit magnitude).
    """
    a = np.frombuffer(raw, dtype=np.int16)
    return (a[0::2].astype(np.float32) + 1j * a[1::2].astype(np.float32)) / SCALE


class ZCDetector:
    """
    Segmented (partially coherent) ZC preamble detector.

    A full-length matched filter correlates coherently over all N samples,
    which gives maximum processing gain but collapses under carrier frequency
    offset: once CFO rotates the phase by more than ~1 cycle across the
    preamble, the peak disappears. This detector instead correlates
    coherently only within blocks of M = N/L samples and sums the block
    magnitudes, tolerating ~L x more CFO at a cost of ~sqrt(L) in SNR gain.

    Choose L from the worst-case CFO: phase should stay roughly constant
    within a block, i.e. df * M / fs << 1.
    """

    def __init__(self, ref, L=8, psr_thresh=8.0):
        """
        Store the reference and pre-split it into the L coherent blocks
        (done once here rather than per buffer, since it never changes).

        ref        : complex ZC reference of length N
        L          : number of blocks; must divide N
        psr_thresh : peak-to-sidelobe ratio required to declare a detection
        """
        self.ref = np.asarray(ref, dtype=complex)
        self.N = len(self.ref)
        if self.N % L:
            raise ValueError(f"L={L} must divide N={self.N}")
        self.L = L
        self.M = self.N // L
        self.psr_thresh = psr_thresh
        self.blocks = [self.ref[l * self.M:(l + 1) * self.M] for l in range(L)]

    def detect(self, y):
        """
        Search y for the preamble and return timing + a coarse CFO estimate.

        Steps:
          1. Correlate each reference block against y. np.correlate conjugates
             its second argument, so entries are sum_n y[j+n]*conj(blk[n]).
          2. Realign: a preamble starting at lag k places block l at j = k+l*M,
             so block l's row is sliced with that offset. After this, column k
             of c holds all L block correlations for the same candidate start.
          3. Sum magnitudes down the columns (non-coherent combining). Discarding
             the phase is exactly what makes the result CFO-robust.
          4. Pick the argmax as the timing estimate, and score it against the
             median of the whole score vector. Median is a robust stand-in for
             the sidelobe/noise level -- a single peak barely shifts it -- so
             the resulting PSR is scale-free and needs no retuning when N or
             the RX gain changes.
          5. Recover CFO from the phase advance between consecutive blocks,
             which are M samples apart.

        Returns dict(found, lag, psr, cfo) with cfo in cycles/sample
        (multiply by fs for Hz).
        """
        y = np.asarray(y, dtype=complex)
        K = len(y) - self.N + 1                 # candidate start lags
        if K < 1:
            return dict(found=False, lag=None, psr=0.0, cfo=0.0)

        # c[l, k] = correlation of block l against y starting at preamble lag k.
        c = np.empty((self.L, K), dtype=complex)
        for l, blk in enumerate(self.blocks):
            full = np.correlate(y, blk, mode="valid")       # len(y)-M+1
            c[l] = full[l * self.M: l * self.M + K]         # realign to lag k

        score = np.abs(c).sum(axis=0)           # non-coherent combining
        k = int(np.argmax(score))
        floor = np.median(score) + 1e-12        # robust sidelobe/noise level
        psr = float(score[k] / floor)

        # Block-to-block phase advance is 2*pi*df*M/fs. Summing the products
        # before taking the angle averages the L-1 pairwise estimates while
        # weighting each by its own correlation strength. Unambiguous only
        # for |df| < fs/(2M); larger offsets alias to a plausible wrong value.
        cfo = 0.0
        if self.L > 1:
            prod = np.sum(c[1:, k] * np.conj(c[:-1, k]))
            cfo = float(np.angle(prod) / (2 * np.pi * self.M))

        return dict(found=psr > self.psr_thresh, lag=k, psr=psr, cfo=cfo)

    def refine(self, y, lag, cfo):
        """
        Post-detection link quality: derotate the located segment by the coarse
        CFO estimate, then run the full-length coherent correlation that
        detect() deliberately avoided.

        Normalizing by both vector norms makes the result invariant to RX gain
        and carrier phase, so it lands in [0,1] with 1.0 = perfect match. This
        is the honest link-quality number, and the one that should visibly
        improve when the 25 ppm crystal is swapped for the ppb-grade OCXO.
        """
        seg = y[lag:lag + self.N] * np.exp(-2j * np.pi * cfo * np.arange(self.N))
        num = np.abs(np.vdot(self.ref, seg))
        den = np.linalg.norm(self.ref) * np.linalg.norm(seg) + 1e-12
        return float(num / den)


def main():
    """
    Open the RX device, then loop forever: refill, unpack, detect, report.

    Each buffer is searched on its own, with no carryover between reads. This
    is only sound because TX repeats the preamble with period N: a contiguous
    read of at least 2N-1 samples then always contains one complete copy no
    matter where the buffer boundary falls, so nothing is lost by treating
    buffers independently. BUF_SAMPLES = 4N satisfies this with margin.

    If TX is ever changed to send a one-shot preamble instead, this no longer
    holds -- a sequence straddling a buffer boundary would be missed, at a rate
    of roughly N/BUF_SAMPLES of arrivals -- and the previous buffer's trailing
    N-1 samples would have to be prepended to restore contiguity.
    """
    ctx = iio.Context(URI)
    rx = ctx.find_device(RX_DEV)
    assert rx is not None, f"missing {RX_DEV}"

    ch_i = rx.find_channel("voltage0", False)  # False = input
    ch_q = rx.find_channel("voltage1", False)
    ch_i.enabled = True
    ch_q.enabled = True

    buf = iio.Buffer(rx, BUF_SAMPLES, False)
    det = ZCDetector(make_zc(N), L=L, psr_thresh=PSR_THRESH)

    while True:
        buf.refill()
        y = from_int16_iq(buf.read())

        r = det.detect(y)
        if r["found"]:
            q = det.refine(y, r["lag"], r["cfo"])
            print(f"ZC @ lag {r['lag']:5d} | PSR {r['psr']:6.1f} | "
                  f"CFO {r['cfo']:+.2e} cyc/sample | coherent match {q:.4f}")
        else:
            print(f"RX: {len(y)} samples, no preamble (PSR {r['psr']:.1f})")


if __name__ == "__main__":
    main()