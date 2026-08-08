
URI_rx      = "ip:192.168.3.1"
URI_tx      = "ip:192.168.2.1"

LO       = int(2.4e9)
FS       = int(10e6)
BW       = int(FS*1.5)
#FS       = BW                   # sample rate (Hz)
OVERSAMPLE = 2                #fs oversampling for fractional 
RX_GAIN     = 71                   # dB, 0 .. 71
TX_GAIN  = -5                  # dB attenuation, range -89.75 .. 0
M        = 2**9        # DOPPLER axis points
N        = 2 ** 10              # DELAY AXIS POINT total samples per frame
T_P      = N                    # TX pulse spacing; sets the delay period M
BUF      =  N*M
PSR_MIN  = 7                 # required peak-to-median in the DD grid
SCALE    = 2 ** 14              # TX is s16; AD9361 takes the top 12 bits
D0_TX       = 3000               # Doppler, in cycles per frame (integer), from -M/2 to M/2-1. The hardware's cyclic replay produces no discontinuity at the wrap. 
D0_IN_HZ = True                # True -> D0 is Hz; see note on cyclic wrap
T0_TX       = 20                  # Delay, in samples, from 0 to N-1. The hardware's cyclic replay produces no discontinuity at the wrap.
T0_IN_US = True                # True -> T0 is in microseconds; see note
PSR_MIN_SIDEPEAKS = 3.2         # required peak-to-median in the DD grid, for the side peaks
PEAK_DB_RANGE = 10             # dB range for the peak plot, relative to the strongest peak
MAG_MIN = 400                 # minimum magnitude for the peak6