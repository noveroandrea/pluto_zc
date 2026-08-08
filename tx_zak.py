"""
TX: transmit a pulse train (pulsone) from a PlutoSDR.

    x[n] = exp(j*2*pi*d0*n/(N*M)) * sum_k delta[n - k*t_p]

A train of impulses spaced t_p samples apart, modulated by a complex
exponential of integer Doppler d0 (in cycles per frame). t_p sets the delay
period, d0 the Doppler shift.
"""

import numpy as np
import adi
import time

from parameters import (URI_tx, LO, BW, FS, TX_GAIN, SCALE,
                        M, N, T_P, D0_TX, D0_IN_HZ, T0_TX, T0_IN_US)



def make_pulsone(n=N*M, d0=D0_TX,t0=T0_TX, t_p=T_P, fs=FS, d0_in_hz=D0_IN_HZ, t0_in_us=T0_IN_US):
    """
    Build one frame of the pulse train.

    The indicator 1(n - k*t_p) is realised by writing the exponential only at
    the pulse positions and leaving the rest zero, rather than by generating
    the full exponential and masking it -- identical result, and it makes the
    sparsity explicit.

    d0 is interpreted as cycles per frame by default, so that the phase is
    periodic over n and the hardware's cyclic replay produces no discontinuity
    at the wrap. Setting d0_in_hz treats it as an absolute frequency, which is
    only seamless when d0/fs happens to be an integer.
    I don't add delay because is already added by poor cfo clock
    """
    if n % t_p:
        raise ValueError(f"t_p={t_p} must divide NM={n}")

    # x = np.zeros(n, dtype=complex)
    # idx = np.arange(0+t0, n, t_p)                  # k*t_p, k = 0 .. n/t_p - 1
    # rate = d0/n if not d0_in_hz else d0/fs #d0 / fs if d0_in_hz else (1/N)*(1-(d0 / M))      # cycles per sample
    # x[idx] = np.exp(2j * np.pi * rate * idx)  # complex exponential at the pulse positions
    # return x


    #nmmatrix=np.zeros((N,M),dtype=complex)

    # # d0_rate = d0 if not d0_in_hz else d0*M/FS  
    # # t0_rate = t0 if not d0_in_us else t0*10**(-6)*FS/N
    # nmmatrix[0,0]=1
    # sending= np.fft.ifft(nmmatrix,axis=1).T.reshape(-1)
    # #normalize the sending signal to have a peak of 1
    # sending=sending/np.max(np.abs(sending))
    # return sending
    
    nmmatrix = np.zeros((N, M), dtype=complex)   # N delays, M Dopplers

    # Doppler bin width is FS/(N*M) Hz; delay bin width is 1/FS seconds.
    # Both must be integers: the grid is indexed, not interpolated, so a
    # requested value that is not an exact bin is silently rounded.
    d_idx = int(round(d0 * N * M / FS)) if d0_in_hz else int(d0)
    t_idx = int(round(t0 * 1e-6 * FS))  if t0_in_us  else int(t0)

    if not -M//2 <= d_idx < M:
        raise ValueError(f"Doppler {d0} -> bin {d_idx}, outside +/-{M//2}")
    if not 0 <= t_idx < N:
        raise ValueError(f"delay {t0} -> bin {t_idx}, outside 0..{N-1}")
    print(f"pulsone: d0={d0} -> bin {d_idx}, t0={t0} -> bin {t_idx}")
    nmmatrix[t_idx, d_idx] = 1
    #nmmatrix[0,0]=1
    sending = np.fft.ifft(nmmatrix, axis=1).T.reshape(-1)
    return sending / np.max(np.abs(sending))


def main():
    print(f"Delay span t_p={N/FS*1e6:.3f} us, Delay resolution {1/FS*1e6:.3f} us, Doppler span v_d={FS/N*1e-3:.1f} KHz, Doppler resolution {FS/(N*M):.1f} Hz")

    sdr = adi.Pluto(URI_tx)

    sdr.sample_rate       = FS
    sdr.tx_lo             = LO
    sdr.tx_rf_bandwidth   = BW
    sdr.tx_hardwaregain_chan0 = TX_GAIN

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

    try:
        t = 0
        while True:
            time.sleep(1)
            t += 1
            print(f"  TX running: {t} s, ~{t*FS/(M*N):.0f} frames sent")
    except KeyboardInterrupt:
        sdr.tx_destroy_buffer()


if __name__ == "__main__":
    main()