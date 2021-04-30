from utilities import IQplot
from constellations import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# === Parameters ===

# Signal to Noise Ratio (SNR), aka Bit Energy (Eb) to Noise Power (N0) in Decibel
EbN0_db = 10


fourPAM = PAM("4-PAM", 4, 8, 0)
sixteenPSK = PSK("16-PSK", 16, 8, 45)
sixteenQAM = QAM("16-QAM", 16, 8, 45)
threeAPSK = APSK("SimpleAPSK", rings=3, symbols_num=[4, 8, 12], radii=[2, 4, 6], angle_offsets=[0, 45, 0])
print(fourPAM.symbols)
fourPAM.plot()
print(sixteenPSK.symbols)
sixteenPSK.plot()
print(sixteenQAM.symbols)
sixteenQAM.plot()
print(threeAPSK.symbols)
threeAPSK.plot()

samplesAPSK = threeAPSK.sampleGenerator(15)
IQplot(samplesAPSK)


# === Grayscale Image Generation - Section III-B ===
# Image resolution (x, y)
img_grid_size = (200, 200)
# I/Q area to map to image
i_range = (-7, 7)
q_range = (-7, 7)
# Samples to be transformed
i_samples = threeAPSK.symbols.real
q_samples = threeAPSK.symbols.imag
# Transform samples from continuous I/Q plane to continuous image plane
y_samples = (i_samples + np.abs(i_range[0])) * img_grid_size[1] / (i_range[1] - i_range[0])
x_samples = - (q_samples - q_range[1]) * img_grid_size[0] / (q_range[1] - q_range[0])
# Quantize samples to pixel values using floor
y_samples = np.floor(y_samples)
x_samples = np.floor(x_samples)
# Clip samples outside the pixel range
# Find samples going over the maximum x value and remove them
x_mask = x_samples >= img_grid_size[0]
x_samples = np.delete(x_samples, x_mask)
y_samples = np.delete(y_samples, x_mask)
# Find samples going over the maximum y value and remove them
y_mask = y_samples >= img_grid_size[1]
x_samples = np.delete(x_samples, y_mask)
y_samples = np.delete(y_samples, y_mask)
# Cast to integers
x_samples = x_samples.astype(int)
y_samples = y_samples.astype(int)
# Number of final samples
samples_num = len(x_samples)
# Numpy array representing number of samples in each pixel value
bin_grid = np.zeros(img_grid_size, dtype='uint16')
# Bin the samples
for i in range(samples_num):
    bin_grid[x_samples[i], y_samples[i]] += 1
# Prepare for grayscale image
# Normalize Grid Array to 255 (8-bit pixel value)
normalized_grid = (bin_grid / np.max(bin_grid)) * 255
# Copy result to uint8 array for writing grayscale image
img_grid = np.empty(img_grid_size, dtype='uint8')
np.copyto(img_grid, normalized_grid, casting='unsafe')
# Generate grayscale image from grid array
img = Image.fromarray(img_grid, mode='L')
# Show Image
# img.show()
# Permanently Save Image
img.save("constellation.jpg")
