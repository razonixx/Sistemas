#------------------------------------------------------------------------------------------------------------------
#   Sample program for PSD analysis.
#------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import math

N = 300         # Window size
fs = 150        # Sampling rate

# Signal with two oscilatory components (5Hz and 40 Hz)
t = [x/fs for x in range(N)]
f = [2*math.cos((2*math.pi/fs)*(5) * x) + 4*math.sin((2*math.pi/fs)*(40) * x) for x in range(N)]

plt.plot(t, f)
plt.show()

# Spectral analysis
power, freq = plt.psd(f, NFFT = N, Fs = fs)
plt.clf()

plt.plot(freq, power)
plt.xlabel('Hz')
plt.ylabel('Power')
plt.show()