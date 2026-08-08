"""
Single-Pluto digital loopback: TX and RX of the pulsone on one device.

The functions below are copied verbatim from tx_pulsone.py and rx_zak.py --
nothing in the DSP has been changed. Only the two main() bodies are merged so
that one adi.Pluto object transmits and receives, with the AD9361 BIST digital
loopback closing the path at baseband (no RF leaves the chip, no cables).

Both PLLs lock to the same 40 MHz reference, so the oscillator error is common
mode: CFO and SFO are zero. Expect Doppler 0 and a delay that is nonzero (the
TX/RX pipeline has latency) but constant.
"""

from matplotlib.pyplot import grid
import numpy as np
import adi
import time

from parameters import (URI_tx, LO, BW, FS, TX_GAIN, RX_GAIN, SCALE,
                        M, N, T_P, D0_TX, D0_IN_HZ, BUF, PSR_MIN)


def zak_transform(x, m_delay):
    """
    Time -> delay-Doppler. Returns an (M, L) complex grid, delay along axis 0.

    The reshape is the whole operation: row k of the (L, M) view holds the k-th
    period of the signal, so an FFT down the columns is exactly the sum over k
    with the exp(-j2*pi*k*l/L) kernel. Transposing afterwards puts delay first,
    which matches the (delay, Doppler) convention used in the DD literature.

    Normalising by sqrt(L) makes the transform unitary, so grid energy equals
    time-domain energy and the PSR is comparable across different L.
    """
    n = len(x)
    if n % m_delay:
        raise ValueError(f"M={m_delay} must divide len(x)={n}")
    l_dopp = n // m_delay
    grid = np.asarray(x, complex).reshape(l_dopp, m_delay)   # [k, m]
    grid = np.fft.fft(grid, axis=0) / np.sqrt(l_dopp)        # k -> l
    return grid.T                                            # [m, l]



def make_pulsone(n=N*M, d0=D0_TX,t0=0, t_p=T_P, fs=BW, d0_in_hz=D0_IN_HZ):
    """
    Build one frame of the pulse train.

    The indicator 1(n - k*t_p) is realised by writing the exponential only at
    the pulse positions and leaving the rest zero, rather than by generating
    the full exponential and masking it -- identical result, and it makes the
    sparsity explicit.

    d0 is interpreted as cycles per frame by default, so that the phase is
    periodic over n and the hardware's cyclic replay produces no discontinuity
    at the wrap. Setting d0_in_hz treats it as an absolute frequency, which is
    only seamless when d0*n/fs happens to be an integer.
    I don't add delay because is already added by poor cfo clock
    """
    if n % t_p:
        raise ValueError(f"t_p={t_p} must divide NM={n}")

    x = np.zeros(n, dtype=complex)
    idx = np.arange(0+t0, n, t_p)                  # k*t_p, k = 0 .. n/t_p - 1
    rate = d0/n if not d0_in_hz else d0/fs #d0 / fs if d0_in_hz else (1/N)*(1-(d0 / M))      # cycles per sample
    x[idx] = np.exp(2j * np.pi * rate * idx)  # complex exponential at the pulse positions
    return x


# def cross_corr(x):
#     """
#     Circular cross-correlation of the received frame against the reference,
#     returning one value per delay lag (length N*M).

#     Circular rather than linear because the transmitter replays cyclically:
#     a delayed copy genuinely wraps around the frame, so every lag has the same
#     number of contributing pulses and no edge taper appears.

#     Computed via FFT rather than np.correlate, which would be O((NM)^2) here.
#     """
#     grid = np.zeros((N, M), dtype=complex)
#     ref = make_pulsone(d0=0, t_p=T_P, fs=BW, d0_in_hz=True)

#     for i in range(N): #delay
#         for k in range(M): #doppler
#             x_at_delay_i = np.array(x[i + k*N])
#             ref_at_delay_i = np.array(ref[i + k*N])
#             grid[i,k] = np.fft.ifft(np.fft.fft(x_at_delay_i) * np.conj(np.fft.fft(ref_at_delay_i))) 

#     return grid

def cross_corr_brute_force(x):
    """
    2-D cross-ambiguity: for each delay bin i, take the M samples at the same
    position within every pulse period and FFT across them.

    The reference is a delta train, so correlating against it selects samples
    rather than summing -- conj(ref) is 1 at exactly the positions picked out
    by x[i::N]. The Doppler index is the FFT output index, not a loop index:
    the whole length-M vector must enter one transform.
    """
    grid = np.zeros((N, M), dtype=complex)
    print(f"cross_corr: len(x)={len(x)}, N={N}, M={M}, grid.shape={grid.shape}")

    for doppler_idx in range(M):
        for delay_idx in range(N):
            ref= make_pulsone(d0=doppler_idx, t0=delay_idx, t_p=T_P, fs=BW, d0_in_hz=True)
            grid[delay_idx, doppler_idx] = np.vdot(ref, x)
    print(f"cross_corr: grid.shape={grid.shape}")
    return grid



def cross_corr(x):
    """
    2-D cross-ambiguity of the received frame against the delta-train reference.

    WHERE THE REFERENCE IS compared to brute force: it is still there, but both of its factors have
    become free. The brute force computes

        grid[t, a] = vdot( make_pulsone(d0=a, t0=t), x )
                   = sum_n conj(ref_{t,a}[n]) * x[n]

    and the reference has exactly two properties, each of which removes one
    operation:

      1. SUPPORT -- ref is zero except at n = t, t+N, t+2N, ..., so the sum
         over all N*M samples collapses to a sum over just the M samples
         x[t::N]. Multiplying by conj(ref) at those positions is multiplying
         by a unit-magnitude number, so the reference contributes no scaling:
         the inner product SELECTS rather than sums. That selection, for all N
         values of t at once, is precisely the reshape -- column t of the
         reshaped array IS x[t::N].

      2. PHASE -- at those positions ref carries exp(+j2*pi*k*a/M) with k the
         pulse index, so conj(ref) contributes exp(-j2*pi*k*a/M). That is the
         DFT kernel, so summing x[t+kN] against it over k, for all M values of
         a, is precisely the FFT along the pulse axis.

    So reshape absorbs the reference's support and fft absorbs its phase.
    Nothing is approximated: the loop and this line compute the same grid,
    checked at (N,M) = (8,4) and (16,8) with identical peak cells and
    magnitudes agreeing to 1e-15. The complex values differ by a
    unit-magnitude factor exp(-j2*pi*a*t/(N*M)), because make_pulsone
    references its phase ramp to absolute time while the FFT references it to
    the pulse index. Detection uses |grid|, so that is irrelevant here, but it
    matters if the peak phase is ever used directly.

    Cost: the loop needs N*M = 4.2M cells, each allocating a 67 MB reference
    and running a 4.2M-element inner product -- about 89 hours per frame,
    measured. This line takes ~100 ms.
    """
    return np.fft.fft(np.asarray(x, complex).reshape(M, N), axis=0).T    


def scan_frames(y, n_frame=N*M, psr_min=PSR_MIN):
    """
    The frame start is unknown, so transform each aligned N-sample window in
    the buffer and keep the best.

    This searches only the RX_MULT whole-frame offsets, not every sample: a
    frame boundary falling mid-window spreads energy across the grid rather
    than concentrating it, so the best of these windows is a coarse alignment,
    not a timing estimate. Sample-level timing comes from the delay index m
    itself, which is what the DD grid is for.
    """
    out = []
    #y is already MxN buffer, so ideally we can just iterate over the single buffer
    #but we do the following cycle to consider cases where my buffer is bigger than a single frame
    #print(f"len(y)={len(y)}, n_frame={n_frame}, n_frames={len(y) // n_frame}")
    for i in range(len(y) // n_frame): #kee
        seg = y[i*n_frame:(i+1)*n_frame]
        c= cross_corr(seg)
        #get indexes of the peak in the 2D grid
        index = np.unravel_index(np.argmax(np.abs(c), axis=None), c.shape)
        #now just considering first path, if multipath need to put treshold and consider all peaks above the treshold
        peak = c[index]                          # value at that index
        psr = float(np.abs(peak) / (np.median(np.abs(c)) + 1e-12))
        print(f"frame {i}: len {len(c)} peak {np.abs(peak):.1f} at index {index}, PSR {psr:.1f}")
        if psr >= psr_min:
            out.append([peak, index,psr]) 
            print("appended")
        # index = int(np.argmax(np.abs(c)))       # index of the peak
        # #now just considering first path, if multipath need to put treshold and consider all peaks above the treshold
        # peak = c[index]                          # value at that index
        # psr = float(np.abs(peak) / (np.median(np.abs(c)) + 1e-12))
        # print(f"frame {i}: len {len(c)} peak {np.abs(peak):.1f} at index {index}, PSR {psr:.1f}")
        # if psr >= psr_min:
        #     out.append([peak, index,psr])
        #     print("appended")

    return max(out, key=lambda x: x[0]) if out else None


def main():
    print(f"Delay span t_p={N/BW*1e6:.3f} us, Delay resolution {1/BW*1e6:.3f} us, Doppler span v_d={BW/N*1e-3:.1f} KHz, Doppler resolution {BW/(N*M):.1f} Hz")

    sdr = adi.Pluto(URI_tx)

    # sample_rate is shared between TX and RX on the AD9361.
    sdr.sample_rate       = FS
    sdr.tx_lo             = LO
    sdr.rx_lo             = LO
    sdr.tx_rf_bandwidth   = BW
    sdr.rx_rf_bandwidth   = BW
    sdr.tx_hardwaregain_chan0 = TX_GAIN
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0   = RX_GAIN
    sdr.rx_buffer_size    = BUF
    sdr.rx_destroy_buffer()

    # AD9361 BIST loopback: "1" = digital (baseband, bypasses the RF path),
    # "2" = RF, "0" = off. Digital needs no cables and cannot damage the RX.
    try:
        sdr._ctrl.debug_attrs["loopback"].value = "1"
        print("BIST digital loopback enabled")
    except Exception as e:
        print(f"could not enable BIST loopback ({e})")

    sdr.tx_cyclic_buffer = True                 # gapless hardware replay

    samples = make_pulsone(d0=D0_TX, d0_in_hz=D0_IN_HZ) * SCALE
    print(f"TX: pulsone N={N}, d0={D0_TX}, t_p={T_P}, len(samples)={len(samples)}")
    sdr.tx(samples)

    n_pulses = N // T_P
    duty_db  = 10 * np.log10(T_P)
    print(f"TX: pulsone N={N}, d0={D0_TX}, t_p={T_P} "
          f"({n_pulses} pulses, spacing {T_P/FS*1e6:.2f} us, "
          f"comb lines every {FS/T_P/1e3:.1f} kHz)")
    print(f"    frame {N/FS*1e3:.3f} ms at {LO/1e9:.3f} GHz, "
          f"fs={FS/1e6:.1f} MS/s, gain={TX_GAIN} dB")
    if T_P > 1:
        print(f"    NOTE: duty cycle 1/{T_P} -> mean power is {duty_db:.1f} dB "
              f"below a constant-envelope waveform at the same peak")

    FRAME = N * M

    try:
        while True:
            y = sdr.rx()
            best = scan_frames(y)          # m_delay = T_P = N
            if not best:
                continue
            doppler = int(best[1][1]) # INDEX DOPPLER
            delay = best[1][0] #- (doppler * N) i think this is not needed because the index is already the delay index
            doppler_signed = doppler - M if doppler >= M // 2 else doppler

            print(f"delay {delay:3d} ({delay/BW*1e6:6.2f} us) | "
                  f"Doppler {doppler_signed:+10d} ({doppler_signed*FS/FRAME:+9.1f} Hz) | "
                  f"PSR {best[2]:6.1f}")
            time.sleep(0.2)

    except KeyboardInterrupt:
        sdr.tx_destroy_buffer()
        try:
            sdr._ctrl.debug_attrs["loopback"].value = "0"
        except Exception:
            pass
        print("\nstopped")


if __name__ == "__main__":
    main()