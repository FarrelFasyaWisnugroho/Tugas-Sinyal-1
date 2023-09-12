print ("Nama : Farrel Fasya.W")
print ("NRP : 5009201066")

import numpy as np
import matplotlib.pyplot as plt

# Generate a noisy signal
t = np.linspace(0, 1, 1000, endpoint=False)  # Time points
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.normal(size=len(t))

# Define a low-pass filter kernel (simple moving average)
filter_length = 11
filter_kernel = np.ones(filter_length) / filter_length

# Perform convolution to apply the low-pass filter
filtered_signal = np.convolve(signal, filter_kernel, mode='same')

# Compute the FFT of the original and filtered signals
fft_signal = np.fft.fft(signal)
fft_filtered_signal = np.fft.fft(filtered_signal)

# Create frequency axis for FFT
freq = np.fft.fftfreq(len(t))

# Plot the original and filtered signals
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.plot(t, signal)
plt.title('Original Signal')
plt.subplot(2, 2, 2)
plt.plot(t, filtered_signal)
plt.title('Filtered Signal')
plt.subplot(2, 2, 3)
plt.plot(freq, np.abs(fft_signal))
plt.title('FFT of Original Signal')
plt.subplot(2, 2, 4)
plt.plot(freq, np.abs(fft_filtered_signal))
plt.title('FFT of Filtered Signal')
plt.tight_layout()
plt.show()
