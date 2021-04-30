import numpy as np
from PIL import Image

# === Grayscale Image Generation - Section III-B ===
def grayscale(symbols, i_range, q_range, img_resolution, filename):
    '''
    Generates Grayscale Image from complex I/Q samples

    :param symbols: Array of complex I/Q samples
    :param i_range: Tuple for I values range to include in image (e.g: (-7,7))
    :param q_range: Tuple for Q values range to include in image (e.g: (-7,7))
    :param img_resolution: Output image resolution
    :param filename: Output image file name
    :return:
    '''
    # Image resolution (x, y)
    # img_resolution = (200, 200)
    # I/Q area to map to image
    # i_range = (-7, 7)
    # q_range = (-7, 7)
    # Samples to be transformed
    i_samples = symbols.real
    q_samples = symbols.imag
    # Transform samples from continuous I/Q plane to continuous image plane
    y_samples = (i_samples + np.abs(i_range[0])) * img_resolution[1] / (i_range[1] - i_range[0])
    x_samples = - (q_samples - q_range[1]) * img_resolution[0] / (q_range[1] - q_range[0])
    # Quantize samples to pixel values using floor
    y_samples = np.floor(y_samples)
    x_samples = np.floor(x_samples)
    # Clip samples outside the pixel range
    # Find samples going over the maximum x value and remove them
    x_mask = x_samples >= img_resolution[0]
    x_samples = np.delete(x_samples, x_mask)
    y_samples = np.delete(y_samples, x_mask)
    # Find samples going over the maximum y value and remove them
    y_mask = y_samples >= img_resolution[1]
    x_samples = np.delete(x_samples, y_mask)
    y_samples = np.delete(y_samples, y_mask)
    # Cast to integers
    x_samples = x_samples.astype(int)
    y_samples = y_samples.astype(int)
    # Number of final samples
    samples_num = len(x_samples)
    # Numpy array representing number of samples in each pixel value
    bin_grid = np.zeros(img_resolution, dtype='uint16')
    # Bin the samples
    for i in range(samples_num):
        bin_grid[x_samples[i], y_samples[i]] += 1
    # Prepare for grayscale image
    # Normalize Grid Array to 255 (8-bit pixel value)
    normalized_grid = (bin_grid / np.max(bin_grid)) * 255
    # Copy result to uint8 array for writing grayscale image
    img_grid = np.empty(img_resolution, dtype='uint8')
    np.copyto(img_grid, normalized_grid, casting='unsafe')
    # Generate grayscale image from grid array
    img = Image.fromarray(img_grid, mode='L')
    # Show Image
    # img.show()
    # Permanently Save Image
    img.save(filename)
