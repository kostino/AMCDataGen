import numpy as np
from utilities import IQplot
from constellations import *
from imgGen import *
import matplotlib.pyplot as plt
from scipy import signal
import math

# Create a QPSK
QPSK = PSK("QPSK", 4, 1, 45)

# Number of symbols and samples per symbol
num_symbols = 100
sps = 8

# Generate random samples. Note: Currently as samples we refer to random symbols.
# Samples in this file are additional samples per symbol for oversampling
qpsk_symbols = QPSK.sampleGenerator(num_symbols)
qpsk_symbols.awgn(SNR=8)

# Plot symbols
qpsk_symbols.plot()

# Seperate real and imaginary parts
i_amps = np.real(qpsk_symbols.samples)
q_amps = np.imag(qpsk_symbols.samples)

# Prepare deltas for pulse shaping
i_pulse_train = np.array([])
q_pulse_train = np.array([])
for index in np.arange(len(qpsk_symbols.samples)):
    # Create array for a pulse
    i_pulse = np.zeros(sps)
    q_pulse = np.zeros(sps)
    # Set the first value to a delta of appropriate height
    i_pulse[0] = i_amps[index]
    q_pulse[0] = q_amps[index]
    # Add samples to signals
    i_pulse_train = np.concatenate((i_pulse, i_pulse_train))
    q_pulse_train = np.concatenate((q_pulse, q_pulse_train))

# Create a raised-cosine filter
num_taps = 101
beta = 0.35
Ts = sps # Assume sample rate is 1 Hz, so sample period is 1, so *symbol* period is 8
t = np.arange(-51, 52) # remember it's not inclusive of final number
h = np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts) / (1 - (2*beta*t/Ts)**2)

# Filter our signal, in order to apply the pulse shaping
i_samples = np.convolve(i_pulse_train, h)
q_samples = np.convolve(q_pulse_train, h)

# Plot I/Q waves
plt.figure(0)
plt.plot(i_samples[40:64], '.-')
plt.plot(q_samples[40:64], '.-')
plt.grid(True)
plt.show()

# Sample waves
sampled_symbols = np.asarray([i_samples[index*sps+num_taps//2+1] + 1j*q_samples[index*sps+num_taps//2+1] for index in np.arange(num_symbols)])
plt.figure(figsize=(5, 5))
plt.scatter(sampled_symbols.real, sampled_symbols.imag)
plt.show()



# === Create and apply fractional delay filter ===
# Fractional delay, in samples
delay = 0.3
# Number of taps
N = 21
n = np.arange(N)
# Calculate delay filter taps
h_delay = np.sinc(n - delay)
# Window the filter to make sure it decays to 0 on both sides
h_delay *= np.hamming(N)
# Normalize to get unity gain, we don't want to change the amplitude/power
h_delay /= np.sum(h_delay)
# Apply filter to I/Q waves
i_samples_frac_delay = np.convolve(i_samples, h_delay)
q_samples_frac_delay = np.convolve(q_samples, h_delay)

# Sample waves
sampled_delayed_symbols = np.asarray([i_samples_frac_delay[index*sps+num_taps//2+1] + 1j*q_samples_frac_delay[index*sps+num_taps//2+1] for index in np.arange(num_symbols)])

# Plot I/Q plane
plt.figure(figsize=(5, 5))
plt.scatter(sampled_delayed_symbols.real, sampled_delayed_symbols.imag)
plt.show()

# Plot I/Q waves
plt.figure(0)
plt.plot(i_samples_frac_delay[40:64], '.-')
plt.plot(q_samples_frac_delay[40:64], '.-')
plt.grid(True)
plt.show()


# === Create and apply a freq offset ===
# Assume our sample rate is 1 MHz
fs = 1e6
# Frequency offset
fo = 10000
# Sampling period
Ts = 1/fs
# Time vector
t = np.arange(0, Ts*len(i_samples), Ts)
# Apply frequency shift
i_samples_freq_shift = i_samples * np.exp(1j*2*np.pi*fo*t)
q_samples_freq_shift = q_samples * np.exp(1j*2*np.pi*fo*t)

# Sample waves
sampled_shifted_symbols = np.asarray([i_samples_freq_shift[index*sps+num_taps//2+1] + 1j*q_samples_freq_shift[index*sps+num_taps//2+1] for index in np.arange(num_symbols)])

# Plot I/Q plane
plt.figure(figsize=(5, 5))
plt.scatter(sampled_shifted_symbols.real, sampled_shifted_symbols.imag)
plt.show()

# Plot I/Q waves
plt.figure(0)
plt.plot(i_samples_freq_shift[40:64], '.-')
plt.plot(q_samples_freq_shift[40:64], '.-')
plt.grid(True)
plt.show()
