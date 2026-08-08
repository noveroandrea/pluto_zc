import numpy as np
from parameters import N, M, T_P, D0_TX, SCALE
from tx_zak import make_pulsone

x = make_pulsone(d0=D0_TX)          # complex, |x| <= 1
iq = (x * SCALE).astype(np.complex64)

# Interleave as int16 I,Q,I,Q... which is what iio_writedev expects.
buf = np.empty(2*len(iq), dtype=np.int16)
buf[0::2] = np.round(iq.real)
buf[1::2] = np.round(iq.imag)
buf.tofile("pulsone.bin")

b = np.fromfile("pulsone.bin", dtype=np.int16)
print(f"bytes      : {b.nbytes}   expected {4*N*M}")
print(f"samples    : {len(b)//2}  expected {N*M}")
print(f"|max|      : {np.abs(b).max()}  (must be < 32767, no clipping)")
print(f"nonzero    : {np.count_nonzero(b[0::2] | b[1::2])} of {N*M}"
      f"  expected {N*M//T_P} pulses")
print(f"first idx  : {np.flatnonzero(b[0::2] | b[1::2])[:4]}  spacing should be {T_P}")