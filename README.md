# pluto_zc

Delay-Doppler (Zak-domain) sounding and preamble synchronisation experiments on
ADALM-Pluto SDRs.

The core experiment transmits a *pulsone* — a single point in the delay-Doppler
(DD) grid, i.e. an impulse train modulated by a complex exponential — replayed
cyclically by the Pluto, and recovers the (delay, Doppler) pair at the receiver
by folding the received frame back into the DD grid. Because the two radios run
off independent 40 MHz references, the recovered peak carries the real clock
error: the delay index drifts with sample-clock offset (SFO) and the Doppler
index sits at the carrier frequency offset (CFO).

Alongside it the repo keeps two earlier synchronisation chains used as
baselines: a Zadoff-Chu matched filter and a LoRa-style up/down-chirp preamble
with joint integer STO/CFO estimation.

## Requirements

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pyadi-iio numpy matplotlib pandas
```

`pyadi-iio` needs libiio installed on the host. A few scripts (`tx.py`,
`rx.py`, `get_pluto_uri.py`) use the raw `iio` Python bindings instead of
`adi`, and target the `iio-emu` emulator rather than real hardware.

`iio-emu/` and `libtinyiiod/` are vendored checkouts of the ADI emulator
tooling, used with `pluto.xml` to replay a fake Pluto context when no radio is
attached.

## Hardware setup

Two Plutos on separate USB-ethernet addresses:

| role | URI |
|------|-----|
| TX   | `ip:192.168.2.1` |
| RX   | `ip:192.168.3.1` |

Set them in `parameters.py` (`URI_tx`, `URI_rx`). `PLUTO_TX_RX_ZAK.PY` instead
runs both ends on a *single* device through the AD9361 BIST digital loopback —
no RF, no cables, and CFO/SFO identically zero because both PLLs share one
reference. That is the sanity check to run first: it should report Doppler 0
and a constant nonzero delay (pipeline latency).

## Configuration

`parameters.py` is the single source of truth for the delay-Doppler chain.
Editing it changes TX and RX together, which matters because the two must agree
on `N`, `M` and `T_P` or the energy does not collapse to one grid point.

| name | default | meaning |
|------|---------|---------|
| `LO` | 2.4 GHz | carrier |
| `FS` | 10 MS/s | sample rate |
| `BW` | `FS*1.5` | analog filter bandwidth |
| `N` | 1024 | delay-axis points (samples per pulse period) |
| `M` | 512 | Doppler-axis points (pulses per frame) |
| `T_P` | `N` | TX pulse spacing; must equal the RX folding period |
| `BUF` | `N*M` | RX buffer = one frame |
| `D0_TX`, `D0_IN_HZ` | 3000, `True` | transmitted Doppler (Hz if `D0_IN_HZ`) |
| `T0_TX`, `T0_IN_US` | 20, `True` | transmitted delay (µs if `T0_IN_US`) |
| `TX_GAIN` / `RX_GAIN` | -5 dB / 71 dB | TX attenuation, RX manual gain |
| `SCALE` | 2^14 | s16 full scale; the AD9361 takes the top 12 bits |
| `PSR_MIN` | 7 | peak-to-median required to declare a detection |
| `OVERSAMPLE` | 2 | RX oversampling factor, fractional-delay path only |

With the defaults, one frame is `N*M` = 524288 samples = 52.4 ms, and the grid
is:

* delay: 0.1 µs resolution, 102.4 µs span (`1/FS`, `N/FS`)
* Doppler: 19.07 Hz resolution, 9.77 kHz span (`FS/(N*M)`, `FS/N`)

The grid is indexed, not interpolated, so `D0_TX`/`T0_TX` are rounded to the
nearest bin: 3000 Hz → Doppler bin 157, 20 µs → delay bin 200.

## Delay-Doppler chain

| file | what it does |
|------|--------------|
| `tx_zak.py` | builds the pulsone by placing a single 1 in the (N, M) DD grid, taking the IFFT along Doppler, and streaming the flattened frame from a cyclic TX buffer |
| `rx_zak.py` | captures one frame, folds it into the DD grid, reports the peak with its PSR, and plots the grid in dB |
| `rx_zak_frac.py` | same, oversampled by `OVERSAMPLE` with per-phase processing, plus multi-peak detection (`PSR_MIN_SIDEPEAKS`, `MAG_MIN`) for multipath |
| `PLUTO_TX_RX_ZAK.PY` | single-device BIST loopback version of the above |
| `make_pulsone_bins.py` | writes the same frame to `pulsone.bin` as interleaved int16 IQ for `iio_writedev`, and verifies length, clipping and pulse spacing |

Run TX and RX in two terminals:

```bash
python tx_zak.py     # streams until Ctrl-C
python rx_zak.py     # captures, prints the peak, shows the DD grid
```

The receiver's cross-correlation against the delta-train reference reduces to
one reshape plus one FFT — `np.fft.fft(x.reshape(M, N), axis=0).T`. The
docstring in `rx_zak.py` derives why (the reference's support becomes the
reshape, its phase becomes the FFT) and records the check against the brute
force double loop: identical peak cells, magnitudes agreeing to 1e-15, and
~100 ms instead of ~89 hours per frame.

## Preamble synchronisation (baselines)

These scripts predate `parameters.py` and keep their own constants at the top
of each file; TX and RX must be edited in step.

| file | what it does |
|------|--------------|
| `tx_preamble.py` | LoRa-style preamble: 6 up-chirps, 2 up-chirps modulated with `S_SYNC`, 6 down-chirps, replayed cyclically |
| `rx_preamble_int.py` | dechirps against up and down references and reads joint integer STO/CFO off the two FFT peaks |
| `tx_basic_corr.py` / `rx_basic_corr.py` | Zadoff-Chu preamble and a segmented matched filter — coherent within `N/L`-sample blocks, magnitudes summed across blocks, tolerating ~L× more CFO for ~sqrt(L) less processing gain |
| `tx.py` / `rx.py` | the same ZC pair written against the raw `iio` API, for the emulator |
| `get_pluto_uri.py` | lists the devices in an iio context |

Dechirping is what makes the chirp preamble CFO-tolerant: the offset moves the
FFT peak instead of cancelling it, so no split into coherent blocks is needed.

## Logs and plots

The committed `rx_log_LO<..>_BW<..>_N<..>_L<..>_<timestamp>.csv` files come from
`rx_preamble_int.py` runs; each carries a `# key=value` header describing the
configuration. (The CSV writing in that script is currently commented out —
`CSV_PATH` is still built, so re-enabling it is a matter of uncommenting the
writer.)

```bash
python plot.py rx_log_*.csv
```

groups the logs by `(N, N_LEN)` and produces four panels: outcome counts,
clean-detection rate with binomial standard error, detection rate over time,
and success rate against `N_LEN` with one line per `N` so the two factors can be
read separately. `log_comparison.png` and `nlen_comparison.png` are committed
examples.

## Notes

* `.gitignore` is a single `.*` rule, so all dotfiles (including `.venv/`) stay
  out of the repo — but `__pycache__/` does not, and a couple of `.pyc` files
  are tracked.
* `rx_zak.py` and `rx_zak_frac.py` are interactive: they block on
  `plt.show()` and, in `rx_zak.py`, break out of the capture loop after the
  first detected frame.
