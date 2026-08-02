"""
RX: LoRa-style joint integer STO/CFO estimation on a PlutoSDR.

Preamble (10 slots of N samples, sent cyclically):
    6 x up-chirp | 2 x up-chirp modulated with symbol S_SYNC | 6 x down-chirp

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

N        = 2**10                # samples per chirp (minimum sampling rate)
S_SYNC   = 200                   # sync-word symbol value
N_UP     = 6                    # unmodulated up-chirps
N_SYNC   = 2                    # sync-word chirps
N_DOWN   = 6                    # down-chirps

N_SLOTS  = 32                   # buffer length in slots; >= 2 x preamble
BUF      = N_SLOTS * N

MIN_RUN  = 5                    # slots of equal bin required to declare a region
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
        b = np.argmax(spec)
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

def slot_peaks(y, ref, n=N):
    """As slot_bins, but also returns the complex FFT value at the peak.
    The magnitude is identical across slots within a region; only the phase
    advances, at 2*pi*eps per slot, and that phase is the fractional CFO."""
    n_slots = len(y) // n
    bins  = np.zeros(n_slots, dtype=int)
    psrs  = np.zeros(n_slots)
    peaks = np.zeros(n_slots, dtype=complex)
    for i in range(n_slots):
        spec = np.fft.fft(y[i*n:(i+1)*n] * ref)
        mag  = np.abs(spec)
        b = int(np.argmax(mag))
        bins[i], psrs[i], peaks[i] = b, mag[b] / (np.median(mag) + 1e-12), spec[b]
    return bins, psrs, peaks


def frac_cfo(peaks, bins, run):
    """
    Fractional CFO in bins from the slot-to-slot phase advance.

    All peaks must be read at the SAME bin index, otherwise the Dirichlet
    kernel contributes a bin-dependent phase and the product is corrupted --
    so BIN_TOL wobble within a run has to be collapsed first. Summing the
    products before taking the angle weights each pair by its own magnitude
    and avoids unwrapping. Unambiguous only over one bin, which is exactly
    the range needed.
    """
    i0, i1 = run
    if i1 - i0 < 2:
        return None
    p = peaks[i0:i1]
    return float(np.angle(np.sum(p[1:] * np.conj(p[:-1]))) / (2 * np.pi))


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


def check_sync(f_sync, f_up, s=S_SYNC, n=N, tol=1):
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
        b_up, p_up, pk_up = slot_peaks(y, dn)
        b_dn, p_dn, pk_dn = slot_peaks(y, up)

        run_up = find_run(b_up, p_up) #run_up is a tuple of (start, stop) indices of the longest run of same bins
        run_dn = find_run(b_dn, p_dn) #run_dn is a tuple of (start, stop) indices of the longest run of same bins

        # frac_up= average(fractional_bin(b_up[run_up[i]:run_up[i+1]]) if run_up is not None else None for i in range(0, len(run_up), 2))
        # frac_dn= average(fractional_bin(b_dn[run_dn[i]:run_dn[i+1]]) if run_dn is not None else None for i in range(0, len(run_dn), 2)) 


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

        # ---- fractional CFO: needs the integer part already resolved ----
        eps_up = frac_cfo(pk_up, b_up, run_up)
        eps_dn = frac_cfo(pk_dn, b_dn, run_dn)
        if eps_up is None or eps_dn is None:
            eps_frac = 0.0
        else:
            # Same oscillator seen twice, so these must agree; averaging on the
            # unit circle avoids the wrap at +/-0.5.
            eps_frac = float(np.angle(np.exp(2j*np.pi*eps_up)
                                    + np.exp(2j*np.pi*eps_dn)) / (2*np.pi))

        cfo_hz = (round(c_bins) + eps_frac) * BW / N

        # Sync slots lie between the two regions.
        msg = ""
        if run_dn[0] > run_up[1]:
            f_s = circ_mean_bin(b_up[run_up[1]:run_dn[0]])
            ok, meas = check_sync(f_s, f_up)
            msg = f" | sync {meas:5.1f} ({'ok' if ok else 'FAIL'})"
            if(ok):
                after 37 slots from the synch symbol get the bin values obtained from downchirp multiplicaiton of the buffer to get the integer cfo and sto
                then get the fractional cfo from the two upchirps like done before and then get the actual cfo in Hz from the integer and fractional cfo.
                (what i expect is that in multipath i will have replicas of those two upchirps in the subsequent slots and i want to get their sto and cfo frac)
                #now read the data slots, each data slot is a double upchirp followed by 10 blank samples
                data_slots = [] #must contain the two upchirps
                multiply for two dowwnchirp reference centered in the middle (shift) (because migth have sto that moves the upchirp window)
                    => like before find CFO and STO intteger, STO minus the shif gives actual sto.  
                
                consider the two upchirps in data_slots  and multiply with two downchirps now non shifted windows and get the cfo fractional.
                then with the cfo fractional and the integer cfo, get the actual cfo in Hz.           

        print(f"up[{run_up[0]}:{run_up[1]}] dn[{run_dn[0]}:{run_dn[1]}] | "
              f"f_up {f_up:6.1f} f_dn {f_dn:6.1f} | "
              f"CFO {cfo_hz:+9.1f} Hz | STO {sto:6.1f}{msg}")


if __name__ == "__main__":
    main()