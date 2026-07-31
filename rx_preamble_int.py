"""
RX: LoRa-style joint integer STO/CFO estimation on a PlutoSDR.

Preamble (10 slots of N samples, sent cyclically):
    4 x up-chirp | 2 x up-chirp modulated with symbol S_SYNC | 4 x down-chirp

Unlike the segmented matched filter, dechirping removes the frequency offset
from the integration itself: CFO relocates the FFT peak instead of cancelling
it. No split into L coherent blocks is therefore needed.
"""

import numpy as np
import adi

URI  = "ip:192.168.3.1"
FS   = int(1e6)
LO   = int(2.4e9)               # 2.4 GHz ISM
BW   = int(1e6)                 # analog filter; here also the chirp bandwidth

N        = 1024                 # samples per chirp (minimum sampling rate)
S_SYNC   = 500                   # sync-word symbol value
N_UP     = 4                    # unmodulated up-chirps
N_SYNC   = 2                    # sync-word chirps
N_DOWN   = 4                    # down-chirps

N_SLOTS  = 32                   # buffer length in slots; >= 2 x preamble
BUF      = N_SLOTS * N

MIN_RUN  = 3                    # slots of equal bin required to declare a region
BIN_TOL  = 1                    # allowed bin wobble within a run
PSR_MIN  = 6.0                  # FFT peak-to-median required per slot


# --------------------------------------------------------------------------
# Waveform
# --------------------------------------------------------------------------

def make_chirp(n, up=True):
    """
    Base chirp, B[k] = exp(j2*pi*(k^2/2N - k/2)).

    The down-chirp is the complex conjugate: the -k/2 term flips sign under
    conjugation, but exp(-j*pi*k) = exp(+j*pi*k) = (-1)^k, so conj(B) is
    exactly the down-sweep of the paper's formulation.
    """
    k = np.arange(n)
    c = np.exp(1j * 2 * np.pi * (k**2 / (2 * n) - k / 2))
    return c if up else np.conj(c)


def make_symbol(n, s, up=True):
    """
    Chirp modulated with symbol s: a cyclic shift of the base chirp.

    Dechirping gives a tone at normalised frequency s/N, which is why an
    integer STO and a symbol value are indistinguishable from one window.
    """
    return np.roll(make_chirp(n, up), -int(s))


def make_preamble(n=N, s=S_SYNC):
    """The full 10N transmit preamble; shared with tx_basic.py."""
    up, dn = make_chirp(n, True), make_chirp(n, False)
    sync = make_symbol(n, s, True)
    return np.concatenate([np.tile(up,   N_UP),
                           np.tile(sync, N_SYNC),
                           np.tile(dn,   N_DOWN)])


# --------------------------------------------------------------------------
# Per-slot dechirping
# --------------------------------------------------------------------------

def slot_bins(y, ref, n=N):
    """
    Split y into N-sample slots on a fixed grid and dechirp each against ref.

    The grid is NOT aligned to the transmitted symbol boundaries -- the offset
    between the two is unknown, and is precisely the integer STO we are trying
    to measure. Alignment is unnecessary because within a homogeneous region
    the signal is periodic with period N, so any contained slot is a cyclic
    shift of the base chirp and still dechirps to a single clean tone.

    Returns (bin index, peak-to-median) per slot.
    """
    n_slots = len(y) // n
    bins = np.zeros(n_slots, dtype=int)
    psrs = np.zeros(n_slots)
    for i in range(n_slots):
        spec = np.abs(np.fft.fft(y[i*n:(i+1)*n] * ref))
        b = int(np.argmax(spec))
        bins[i] = b
        psrs[i] = spec[b] / (np.median(spec) + 1e-12)
    return bins, psrs


def _circ_dist(a, b, n=N):
    """Distance between two bin indices, accounting for FFT wraparound."""
    d = abs(int(a) - int(b)) % n
    return min(d, n - d)


def find_run(bins, psrs, min_run=MIN_RUN, tol=BIN_TOL, psr_min=PSR_MIN):
    """
    Longest run of consecutive slots returning the same bin with a strong peak.

    This run IS the detection event. A single slot's bin could be anything;
    what identifies a chirp region is that consecutive slots return an
    *identical* bin, which happens only because the chirps are identical and
    the offsets are constant across them.

    Returns (start, stop) slot indices, or None.
    """
    best, i, n = None, 0, len(bins)
    while i < n:
        if psrs[i] < psr_min:
            i += 1
            continue
        j = i + 1
        while (j < n and psrs[j] >= psr_min
               and _circ_dist(bins[j], bins[i]) <= tol):
            j += 1
        if j - i >= min_run and (best is None or j - i > best[1] - best[0]):
            best = (i, j)
        i = max(j, i + 1)
    return best


def circ_mean_bin(b, n=N):
    """
    Circular mean of bin indices: plain averaging fails when the run straddles
    the wraparound (e.g. bins 1023, 0, 1 would average to ~341).
    """
    z = np.exp(2j * np.pi * np.asarray(b) / n).mean()
    return (np.angle(z) / (2 * np.pi) * n) % n


# --------------------------------------------------------------------------
# Joint solution
# --------------------------------------------------------------------------

def solve_sto_cfo(f_up, f_dn, n=N):
    """
    Recover integer CFO and STO from the two measured bins.

    With c = N*CFO/BW and t = (N - STO) mod N,
        f_up = c + t  (mod N),      f_dn = c - t  (mod N),
    so  2c = f_up + f_dn (mod N). Dividing by 2 modulo N admits two solutions
    separated by N/2 -- this is the modulo-BW/2 ambiguity of the method. Once c
    is chosen, t = f_up - c follows uniquely, so there are exactly two
    candidate (CFO, STO) pairs, not four.

    The ambiguity is resolved by the physical bound |CFO| < BW/4, i.e.
    |c| < N/4: the two candidates differ by N/2, so at most one can satisfy it.
    Note the sync word does NOT resolve this -- see check_sync().
    """
    ssum = (f_up + f_dn) % n
    out = []
    for half in (0.0, n / 2.0):
        c = (ssum / 2.0 + half) % n
        c_signed = c - n if c >= n / 2 else c          # wrap to [-N/2, N/2)
        t = (f_up - c) % n
        sto = (n - t) % n
        out.append((c_signed, sto))
    # Keep the candidate inside the recoverable CFO range.
    inside = [p for p in out if abs(p[0]) < n / 4]
    return (inside[0] if len(inside) == 1 else None), out


def check_sync(f_sync, f_up, s=S_SYNC, n=N, tol=2):
    """
    Consistency check: the sync chirps carry symbol s on top of the same STO
    and CFO, so their bin sits s above the unmodulated up-chirp bin:

        (f_sync - f_up) mod N = s

    Both the CFO and STO terms cancel in the difference, so this validates
    detection and catches a bin-index error modulo N -- but, being independent
    of how (c, t) was split, it cannot resolve the N/2 ambiguity above.
    """
    meas = (f_sync - f_up) % n
    return _circ_dist(meas, s, n) <= tol, meas


# --------------------------------------------------------------------------

def main():
    sdr = adi.Pluto(URI)
    sdr.sample_rate     = FS
    sdr.rx_lo           = LO
    sdr.rx_rf_bandwidth = BW
    sdr.rx_buffer_size  = BUF
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0   = 40
    sdr.rx_destroy_buffer()

    up, dn = make_chirp(N, True), make_chirp(N, False)

    print(f"RX: fs={FS/1e6:.1f} MS/s, bin = {BW/N:.1f} Hz, "
          f"recoverable CFO +/-{BW/4/1e3:.1f} kHz")

    while True:
        y = sdr.rx()

        # Up-chirps are dechirped by the DOWN reference and vice versa.
        b_up, p_up = slot_bins(y, dn)
        b_dn, p_dn = slot_bins(y, up)

        run_up = find_run(b_up, p_up)
        run_dn = find_run(b_dn, p_dn)
        if run_up is None or run_dn is None or run_dn[0] < run_up[1]:
            print("no preamble")
            continue

        f_up = circ_mean_bin(b_up[run_up[0]:run_up[1]])
        f_dn = circ_mean_bin(b_dn[run_dn[0]:run_dn[1]])

        pick, cands = solve_sto_cfo(f_up, f_dn)
        if pick is None:
            print(f"ambiguous: f_up={f_up:.1f} f_dn={f_dn:.1f} "
                  f"candidates={[(round(c,1), round(s,1)) for c, s in cands]}")
            continue
        c_bins, sto = pick
        cfo_hz = c_bins * BW / N

        # Sync slots lie between the two regions.
        msg = ""
        if run_dn[0] > run_up[1]:
            f_s = circ_mean_bin(b_up[run_up[1]:run_dn[0]])
            ok, meas = check_sync(f_s, f_up)
            msg = f" | sync {meas:5.1f} ({'ok' if ok else 'FAIL'})"

        print(f"up[{run_up[0]}:{run_up[1]}] dn[{run_dn[0]}:{run_dn[1]}] | "
              f"f_up {f_up:6.1f} f_dn {f_dn:6.1f} | "
              f"CFO {cfo_hz:+9.1f} Hz | STO {sto:6.1f}{msg}")


if __name__ == "__main__":
    main()