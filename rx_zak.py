"""
RX: delay-Doppler processing of the received pulsone on a PlutoSDR.

The received frame is folded into the delay-Doppler grid: for each delay bin
the M samples at the same position within every pulse period are collected and
FFT'd across pulses. The peak cell gives (delay, Doppler) jointly.
"""

from matplotlib.pyplot import grid
import numpy as np
import adi
import matplotlib.pyplot as plt
import time

from parameters import (URI_rx, LO, BW, FS, RX_GAIN,
                        M, N, T_P, D0_TX, D0_IN_HZ, BUF, PSR_MIN)



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
        # #now just considering first path, if multipath need to put treshold and consider all peaks above the treshold
        peak = c[index]                          # value at that index
        psr = float(np.abs(peak) / (np.median(np.abs(c)) + 1e-12))
        print(f"frame {i}: len {len(c)} peak {np.abs(peak):.1f} at index {index}, PSR {psr:.1f}")
        if psr >= psr_min:
        #    out.append([peak, index,psr]) 
            print("appended")
            out.append([peak, index, c]) 

    return out if out else None #max(out, key=lambda x: x[0]) if out else None


def main():
    print(f"Delay span t_p={N/BW*1e6:.3f} us, Delay resolution {1/BW*1e6:.3f} us, Doppler span v_d={BW/N*1e-3:.1f} KHz, Doppler resolution {BW/(N*M):.1f} Hz")

    sdr = adi.Pluto(URI_rx)
    sdr.sample_rate       = FS
    sdr.rx_lo             = LO
    sdr.rx_rf_bandwidth   = BW
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0   = RX_GAIN
    sdr.rx_buffer_size    = BUF
    sdr.rx_destroy_buffer()

    FRAME = N * M

    try:
        while True:
            y = sdr.rx()
            corr_values = scan_frames(y)
            if not corr_values:
                continue
            peak, index, grid = corr_values[0]

            mag = np.abs(grid)
            db  = 20*np.log10(mag / mag.max() + 1e-12)
            db  = np.fft.fftshift(db, axes=1)     # zero Doppler to the centre

            plt.figure(figsize=(10, 5))
            plt.imshow(db, aspect='auto', origin='lower', vmin=-40, vmax=0,
                       extent=[-M/2*FS/(N*M), M/2*FS/(N*M),   # Doppler, Hz
                               0, N/FS*1e6])                   # delay, us
            plt.colorbar(label='dB below peak')
            plt.title(f"Delay-Doppler | peak at delay {index[0]}, Doppler {index[1]}")
            delay=index[0]
            doppler=index[1]
            doppler_signed = doppler - M if doppler >= M // 2 else doppler
            print(f"delay {delay:3d} ({delay/BW*1e6:6.2f} us) | "
                   f"Doppler {doppler_signed:+10d} ({doppler_signed*FS/FRAME:+9.1f} Hz) | ")
            plt.xlabel("Doppler (Hz)")
            plt.ylabel("Delay (us)")
            plt.tight_layout()
            plt.show()
            break

            # if not best:
            #     continue
            # doppler = int(best[1][1]) # INDEX DOPPLER
            # delay = best[1][0] #- (doppler * N) i think this is not needed because the index is already the delay index

            # doppler_signed = doppler - M if doppler >= M // 2 else doppler

            # print(f"delay {delay:3d} ({delay/BW*1e6:6.2f} us) | "
            #       f"Doppler {doppler_signed:+10d} ({doppler_signed*FS/FRAME:+9.1f} Hz) | "
            #       f"PSR {best[2]:6.1f}")
            # time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()