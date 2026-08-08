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
import csv, time, datetime
import matplotlib.pyplot as plt


URI  = "ip:192.168.3.1"
LO   = int(2.4e9)               # 2.4 GHz ISM
FS   = int(10e6)                 # analog filter; here also the chirp bandwidth
BW= FS*1.5                       # sample rate

N        = 2**10                # samples per chirp (minimum sampling rate)
S_SYNC   = 200                   # sync-word symbol value
N_LEN     = 6                    # unmodulated up-chirps and downchirps
N_SYNC   = 2                    # sync-word chirps
MIN_RUN  = N_LEN-1                    # slots of equal bin required to declare a region



N_UP,N_DOWN = N_LEN,N_LEN

N_SLOTS  = 3*(N_UP + N_SYNC + N_DOWN)                   # buffer length in slots; >= 2 x preamble
BUF      = N_SLOTS * N

BIN_TOL  = 0                    # allowed bin wobble within a run
PSR_MIN  = 70.0                  # FFT peak-to-median required per slot


CSV_PATH = (f"rx_log_LO{LO//10**6}M_BW{BW//10**6}M_N{N}_L{N_LEN}"
            f"_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv")

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
               and _circ_dist(bins[j], bins[i]) <= tol and j - i < N_LEN - 1):  #need to check only the first N_LEN-1 slots, because the last slot is not fully contained in the region
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




def setup_plot(n, fs):
    """
    Figure built once; only artist DATA is replaced afterwards. The title and
    axis labels are set here since they never change.
    """
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))

    extent = [-fs/2, fs/2,          # CFO, Hz  (n bins of fs/n)
              0, n / fs * 1e6]       # STO, us  (n samples of 1/fs)

    im = ax.imshow(np.full((n, n), -40.0), aspect='auto', origin='lower',
                   vmin=-40, vmax=0, extent=extent)
    mk, = ax.plot([], [], 'o', mfc='none', mec='red', ms=10, mew=1.5)
    ax.set_xlabel("CFO (Hz)")
    ax.set_ylabel("STO (us)")
    ax.set_title("STO-CFO map")
    fig.colorbar(im, ax=ax, label='dB below peak')
    fig.tight_layout()
    return fig, ax, im, mk


def update_plot(fig, im, mk, score, cfo_axis, peaks, n, fs):
    """
    Push new data into the existing artists.

    The CFO axis is rolled so negative frequencies sit on the left, matching
    the extent given at setup; the same signed conversion is applied to the
    peak coordinates so the markers land on the cells they describe.
    """
    db = 20*np.log10(score / (score.max() + 1e-12) + 1e-12)
    im.set_data(np.roll(db, n//2, axis=1))

    bin_hz = fs / n
    mk.set_data([cfo_axis[c] * bin_hz for _, c in peaks],
                [s / fs * 1e6 for s, _ in peaks])

    fig.canvas.draw_idle()
    plt.pause(0.001)

def find_peaks(mag, psr_min=PSR_MIN):
    """
    All cells whose peak-to-median ratio exceeds psr_min.

    The median over the whole grid estimates the noise floor: only a few of
    the N*M cells hold signal, so the median is unaffected by them. Using a
    ratio rather than an absolute level keeps the threshold independent of RX
    gain and range.

    Returns [[delay_idx, doppler_idx], ...], strongest first.
    """
    floor = np.median(mag) + 1e-12
    ms, ls = np.nonzero(mag > psr_min * floor)
    order = np.argsort(-mag[ms, ls])
    return [[int(ms[i]), int(ls[i])] for i in order]


def region_spectrum(y, ref, run, n=N):
    """
    Magnitude spectrum averaged over the slots of one homogeneous region.

    Averaging the magnitudes (not the complex values) across the run is the
    same non-coherent combining used for detection: it raises the peaks
    relative to the noise floor without needing the slot-to-slot phase, which
    the fractional estimator uses separately.
    """
    i0, i1 = run
    acc = np.zeros(n)
    for i in range(i0, i1):
        acc += np.abs(np.fft.fft(y[i*n:(i+1)*n] * ref))
    return acc / (i1 - i0)


def sto_cfo_map(s_up, s_dn, n=N):
    """
    Joint STO-CFO surface: the chirp-system analogue of a delay-Doppler grid.

    Each dechirped spectrum alone gives only a sum of the two unknowns,
        f_up = c + t  (mod n),   f_dn = c - t  (mod n),
    with c the CFO in bins and t = (n - STO) mod n. Scoring every hypothesis
    (c, t) by
        score = |S_up[c+t]| * |S_dn[c-t]|
    makes the separation explicit: a true path lights up both spectra at the
    consistent pair of bins, so the product is large, while a spurious
    combination is large in at most one factor.

    Multipath appears as several bright cells -- but so do cross-pairings, an
    up-peak of one path with a down-peak of another. With P and Q peaks the
    surface has P*Q maxima and only min(P,Q) real paths, so cells are
    candidates rather than detections.

    Returns (score, sto_axis, cfo_axis); score is indexed [sto, cfo] and
    cfo_axis is signed.
    """
    sto = np.arange(n)                        # delay, in samples
    c = np.arange(n)                          # CFO, in bins (unsigned)
    t = (n - sto) % n                         # the quantity the equations use

    T, C = np.meshgrid(t, c, indexing='ij')   # [sto, cfo]
    score = s_up[(C + T) % n] * s_dn[(C - T) % n]

    cfo_axis = np.where(c >= n//2, c - n, c)  # wrap to [-n/2, n/2)
    return score, sto, cfo_axis

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


    # One file per run, named by start time, so successive experiments never
    # overwrite each other. The config is written as comment lines rather than
    # repeated per row: pandas.read_csv(path, comment='#') skips them, and it
    # keeps each capture self-describing when several are compared later.
    # f_csv = open(CSV_PATH, "w", newline="")
    # for k, v in [("LO_Hz", LO), ("BW_Hz", BW), ("FS_Hz", FS), ("N", N),
    #              ("S_SYNC", S_SYNC), ("N_UP", N_UP), ("N_SYNC", N_SYNC),
    #              ("N_DOWN", N_DOWN), ("N_SLOTS", N_SLOTS),
    #              ("MIN_RUN", MIN_RUN), ("BIN_TOL", BIN_TOL),
    #              ("PSR_MIN", PSR_MIN), ("RX_GAIN_dB", 40),
    #              ("bin_Hz", BW / N), ("sample_us", 1e6 / FS)]:
    #     f_csv.write(f"# {k}={v}\n")

    # w = csv.writer(f_csv)
    # w.writerow(["t_s", "run_up0", "run_up1", "run_dn0", "run_dn1",
    #             "f_up", "f_dn", "c_bins", "tie",
    #             "cfo_int_hz", "cfo_frac_hz", "cfo_hz",
    #             "eps_up", "eps_dn", "eps_frac",
    #             "sto_bins", "sto_us", "sync_meas", "sync_ok"])
    # f_csv.flush()
    t0 = time.monotonic()



    print(f"RX: fs={FS/1e6:.1f} MS/s, bin = {BW/N:.1f} Hz, "
          f"recoverable CFO +/-{BW/4/1e3:.1f} kHz")
    fig, ax, im, mk = setup_plot(N, FS)      # the map version, before the loop

    try: 
        while True:
            y = sdr.rx()

            # Up-chirps are dechirped by the DOWN reference and vice versa.
            b_up, p_up, pk_up = slot_peaks(y, dn)
            b_dn, p_dn, pk_dn = slot_peaks(y, up)

            run_up = find_run(b_up, p_up)
            run_dn = find_run(b_dn, p_dn)

            if run_up is None or run_dn is None or run_dn[0] < run_up[1]:
                print("no preamble")
                continue

            f_up = circ_mean_bin(b_up[run_up[0]:run_up[1]])
            f_dn = circ_mean_bin(b_dn[run_dn[0]:run_dn[1]])

            pick, cands = solve_sto_cfo(f_up, f_dn)
            if pick is None:
                print(f"ambiguous: f_up={f_up:.1f} f_dn={f_dn:.1f}")
                continue
            c_bins, sto = pick
            sto_us = sto / FS * 1e6

            # ---- fractional CFO ----
            eps_up = frac_cfo(pk_up, b_up, run_up)
            eps_dn = frac_cfo(pk_dn, b_dn, run_dn)
            if eps_up is None and eps_dn is None:
                eps_frac = 0.0
            elif eps_dn is None:
                eps_frac = eps_up
            elif eps_up is None:
                eps_frac = -eps_dn
            else:
                eps_frac = float(np.angle(np.exp(2j*np.pi*eps_up)
                                        + np.exp(-2j*np.pi*eps_dn)) / (2*np.pi))

            n_int = int(np.floor(c_bins))
            if (c_bins - n_int) > 0.25 and eps_frac < 0:
                n_int += 1
            bin_hz = FS / N
            cfo_hz = (n_int + eps_frac) * bin_hz

            # ---- STO-CFO map ----
            s_up = region_spectrum(y, dn, run_up, N)
            s_dn = region_spectrum(y, up, run_dn, N)
            score, sto_axis, cfo_axis = sto_cfo_map(s_up, s_dn, N)
            peaks = find_peaks(score, psr_min=PSR_MIN)

            update_plot(fig, im, mk, score, cfo_axis, peaks, N, FS)

            continue # skip the CSV logging for now, because it is not needed for the map version
            # ---- fractional CFO ----
            # The down-chirp region is dechirped with the conjugate reference, so
            # its slot-to-slot phase advance carries the OPPOSITE sign: negate it
            # before combining, or the two estimates cancel instead of averaging.
            eps_up = frac_cfo(pk_up, b_up, run_up)
            eps_dn = frac_cfo(pk_dn, b_dn, run_dn)
            if eps_up is None and eps_dn is None:
                eps_frac = 0.0
            elif eps_dn is None:
                eps_frac = eps_up
            elif eps_up is None:
                eps_frac = -eps_dn
            else:
                eps_frac = float(np.angle(np.exp(2j*np.pi*eps_up)
                                        + np.exp(-2j*np.pi*eps_dn)) / (2*np.pi))

            # ---- combine integer and fractional parts ----
            # c_bins can land on a half-integer, because solve_sto_cfo divides the
            # bin sum by 2. That .5 is an arithmetic remainder, not a measurement:
            # every argmax is an integer and BIN_TOL = 0 forces the run to agree,
            # so the only real sub-bin information is in eps_frac. floor is used
            # rather than round because round() is banker's -- 100.5 -> 100 but
            # 101.5 -> 102 -- so the tie would break in a direction that alternates
            # with bin parity. With floor the residue is always 0.0 or 0.5, and a
            # negative eps_frac on a .5 residue means the true value sits above the
            # midpoint, so the integer part is the upper bin.
            n_int = int(np.floor(c_bins))
            tie = (c_bins - n_int) > 0.25
            if tie and eps_frac < 0:
                n_int += 1

            bin_hz   = BW / N
            cfo_int  = n_int * bin_hz
            cfo_frac = eps_frac * bin_hz
            cfo_hz   = cfo_int + cfo_frac
            bin_tot= n_int + eps_frac

            # ---- sync-word consistency check ----
            # Trims the straddling slot at run_up[1]; with N_SYNC = 2 the single
            # remaining slot is the only fully contained one.
            # msg = ""
            # if run_dn[0] > run_up[1] + 1:
            #     f_s = circ_mean_bin(b_up[run_up[1]+1:run_dn[0]])
            #     ok, meas = check_sync(f_s, f_up)
            #     msg = f" | sync {meas:5.1f} ({'ok' if ok else 'FAIL'})"




            # sync_meas, sync_ok = "", ""
            # msg = ""
            # if run_dn[0] > run_up[1] + 1:
            #     f_s = circ_mean_bin(b_up[run_up[1]+1:run_dn[0]])
            #     ok, meas = check_sync(f_s, f_up)
            #     sync_meas, sync_ok = f"{meas:.3f}", int(ok)
            #     msg = f" | sync {meas:5.1f} ({'ok' if ok else 'FAIL'})"

            # w.writerow([f"{time.monotonic()-t0:.3f}",
            #             run_up[0], run_up[1], run_dn[0], run_dn[1],
            #             f"{f_up:.3f}", f"{f_dn:.3f}", f"{c_bins:.3f}", int(tie),
            #             f"{cfo_int:.3f}", f"{cfo_frac:.3f}", f"{cfo_hz:.3f}",
            #             "" if eps_up is None else f"{eps_up:.6f}",
            #             "" if eps_dn is None else f"{eps_dn:.6f}",
            #             f"{eps_frac:.6f}",
            #             f"{sto:.3f}", f"{sto_us:.4f}", sync_meas, sync_ok])
            # f_csv.flush()   # so the file is usable while the capture is still running

            # print(f"up[{run_up[0]}:{run_up[1]}] dn[{run_dn[0]}:{run_dn[1]}] | "
            #     f"f_up {f_up:6.1f} f_dn {f_dn:6.1f}{'  TIE' if tie else ''} | "
            #     f"CFO bins= {n_int:6.1f} {eps_frac:+7.7f} = {bin_tot:9.7f} | "
            #     f"CFO Hz= {cfo_int:+9f} {cfo_frac:+7f} = {cfo_hz:+9f} Hz | "
            #     f"STO {sto_us:8.3f} us{msg}")
    except KeyboardInterrupt:
         pass
    # finally:
    #     f_csv.close()
    #     print(f"RX: done, log written to {CSV_PATH}")



if __name__ == "__main__":
    main()
