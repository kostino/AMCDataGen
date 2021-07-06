import numpy as np
from utilities import IQplot
from constellations import *
from imgGen import *
import matplotlib.pyplot as plt
from scipy import signal
import math

# Modulation parameters
# sps = 8
Fs = int(6e4)           # the sampling frequency we use for the discrete simulation of analog signals
fc = int(3e3)           # 3kHz carrier frequency
Ts = 1e-3               # 1 ms symbol spacing, i.e. the baseband samples are Ts seconds apart.
BN = 1/(2*Ts)           # the Nyquist bandwidth of the baseband signal.
ups = int(Ts*Fs)        # number of samples per symbol in the "analog" domain
num_symbols = 10        # number of transmitted baseband samples
t0=Ts

# Create a QPSK
QPSK = PSK("QPSK", 4, 1, 45)

# Generate random samples. Note: Currently as samples we refer to random symbols.
# Samples in this file are additional samples per symbol for oversampling
qpsk_symbols = QPSK.sampleGenerator(num_symbols)
qpsk_symbols.awgn(SNR=8)

# Plot symbols
qpsk_symbols.plot()

# Time instants of baseband samples
t_symbols = Ts * np.arange(num_symbols)

# Create dirac delta series for pulse shaping
x = np.zeros(ups*num_symbols, dtype='complex')
x[::ups] = qpsk_symbols.samples

# Time instants of steps per symbol
t_x = np.arange(len(x))/Fs

plt.figure(1)
plt.figure(figsize=(8,3))
plt.subplot(121)
plt.plot(t_x/Ts, x.real)
plt.subplot(122)
plt.plot(t_x/Ts, x.imag)
plt.show()

# Create a raised-cosine filter
num_taps = 101
beta = 0.35
Ts_f = ups # Assume sample rate is 1 Hz, so sample period is 1, so *symbol* period is 8
t = np.arange(-51, 52) # remember it's not inclusive of final number
h = np.sinc(t/Ts_f) * np.cos(np.pi*beta*t/Ts_f) / (1 - (2*beta*t/Ts_f)**2)

# Convolve
u = np.convolve(x, h)
# Time instants of convolution results on a step level
t_u = np.arange(len(u))/Fs

plt.figure(2)
plt.subplot(121)
plt.plot((t_x+t0)/Ts, x.real, label='$x(t)$') # artificial extra delay for the baseband samples
plt.plot(t_u/Ts, u.real, label='$u(t)$')
plt.subplot(122)
plt.plot((t_x+t0)/Ts, x.imag)
plt.plot(t_u/Ts, u.imag)
plt.show()

i = u.real
q = u.imag

iup = i * np.cos(2*np.pi*t_u*fc)
qup = q * -np.sin(2*np.pi*t_u*fc)

# Plot the time-domain signals
plt.figure(3)
plt.subplot(211)
plt.plot(t_u/Ts, iup, label='$i_{up}(t)$')
plt.plot(t_u/Ts, i, 'r', label='$i(t)$')

plt.subplot(212)
plt.plot(t_u/Ts, qup, label='$q_{up}(t)$')
plt.plot(t_u/Ts, q, 'r', label='$q(t)$')
plt.show()

s = iup + qup

idown = s * np.cos(2*np.pi*-fc*t_u)
qdown = s * -np.sin(2*np.pi*fc*t_u)

# Plot the time-domain signals
plt.figure(3)
plt.subplot(211)
plt.plot(t_u/Ts, idown, label='$i_{up}(t)$')
plt.plot(t_u/Ts, i, 'r', label='$i(t)$')

plt.subplot(212)
plt.plot(t_u/Ts, qdown, label='$q_{up}(t)$')
plt.plot(t_u/Ts, q, 'r', label='$q(t)$')
plt.show()

# Low-pass filter
cutoff = 5*BN        # arbitrary design parameters
lowpass_order = 51
lowpass_delay = (lowpass_order // 2)/Fs  # a lowpass of order N delays the signal by N/2 samples (see plot)
# design the filter
lowpass = signal.firwin(lowpass_order, cutoff/(Fs/2))

# calculate frequency response of filter
t_lp = np.arange(len(lowpass))/Fs
f_lp = np.linspace(-Fs/2, Fs/2, 2048, endpoint=False)
H = np.fft.fftshift(np.fft.fft(lowpass, 2048))

plt.figure(4)
plt.subplot(121)
plt.plot(t_lp/Ts, lowpass)
plt.gca().annotate(r'$\tau_{LP}$', xy=(lowpass_delay/Ts,0.08), xytext=(lowpass_delay/Ts+0.3, 0.08), arrowprops=dict(arrowstyle='->'))
plt.subplot(122)
plt.plot(f_lp, 20*np.log10(abs(H)))
plt.show()