import numpy as np
from PIL import Image


# === Function to map from I/Q plane to X-Y image plane ===
def IQtoXY(symbols, i_range, q_range, img_resolution):
    """
    Maps complex I/Q samples to an X-Y plane compatible with 2D array indices

    :param symbols: Complex I/Q samples array
    :param i_range: Tuple indicating the range of I to be converted e.g: (-7,7)
    :param q_range: Tuple indicating the range of Q to be converted e.g: (-7,7)
    :param img_resolution: Target X-Y plane dimensions e.g: (200,200) creates a 200x200 image
    :return: x_samples, y_samples: X and Y coordinates of converted samples
    """
    # Samples to be transformed
    i_samples = symbols.real
    q_samples = symbols.imag
    # Transform samples from continuous I/Q plane to continuous image plane
    y_samples = (i_samples + np.abs(i_range[0])) * img_resolution[1] / (i_range[1] - i_range[0])
    x_samples = - (q_samples - q_range[1]) * img_resolution[0] / (q_range[1] - q_range[0])
    # Clip samples outside the pixel range
    # Find samples going over the maximum x value and remove them
    x_mask = x_samples >= img_resolution[0]
    x_samples = np.delete(x_samples, x_mask)
    y_samples = np.delete(y_samples, x_mask)
    # Find samples going over the maximum y value and remove them
    y_mask = y_samples >= img_resolution[1]
    x_samples = np.delete(x_samples, y_mask)
    y_samples = np.delete(y_samples, y_mask)
    # Return X and Y coordinates of converted samples
    return x_samples, y_samples


# === Grayscale Image Generation - Section III-B ===
def grayscaleImgGen(symbols, i_range, q_range, img_resolution, filename):
    """
    Generates Grayscale Image from complex I/Q samples

    :param symbols: Array of complex I/Q samples
    :param i_range: Tuple for I values range to include in image (e.g: (-7,7))
    :param q_range: Tuple for Q values range to include in image (e.g: (-7,7))
    :param img_resolution: Output image resolution (x,y) (e.g: (200,200))
    :param filename: Output image file name
    :return:
    """

    # Transform I/Q samples to XY plane
    x_samples, y_samples = IQtoXY(symbols, i_range, q_range, img_resolution)
    # Quantize samples to pixel values using floor
    y_samples = np.floor(y_samples)
    x_samples = np.floor(x_samples)
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


# === Enhanced Grayscale and RGB Image Generation - Section III-C&D ===
def enhancedImgGen(symbols, i_range, q_range, img_resolution, filename, channels, power, decay):
    """
    Generates Enhanced Grayscale and RGB Images from complex I/Q samples using exponential decay.

    :param symbols: Array of complex I/Q samples
    :param i_range: Tuple for I values range to include in image (e.g: (-7,7))
    :param q_range: Tuple for Q values range to include in image (e.g: (-7,7))
    :param img_resolution: Output image resolution (x,y) (e.g: (200,200))
    :param filename: Output image file name
    :param channels: Number of image channels: 1 -> Grayscale, 3 -> RGB
    :param power: Tuple for power of I/Q samples on each layer/channel
    :param decay: Tuple for exponential decay coefficient for each layer/channel
    :return:
    """
    # Transform I/Q samples to XY plane
    x_samples, y_samples = IQtoXY(symbols, i_range, q_range, img_resolution)

    # Number of final samples
    samples_num = len(x_samples)

    # Add channel number to dimensions if dimensions > 1
    if channels > 1:
        img_resolution = (img_resolution[0], img_resolution[1], channels)

    # Numpy array representing the 'power' of each pixel value as influenced by each sample
    power_grid = np.zeros(img_resolution, dtype='float64')

    # Calculate pixel centroids in continuous x,y plane
    x_centroids = np.arange(start=0.5, stop=img_resolution[0], step=1, dtype='float32').reshape((img_resolution[0], 1))
    y_centroids = np.arange(start=0.5, stop=img_resolution[1], step=1, dtype='float32').reshape((1, img_resolution[1]))

    centroid_distances = np.zeros((img_resolution[0], img_resolution[1], samples_num))
    for sample in range(samples_num):
        # Hacky optimization to skip calculations. Cuts significant time
        # if abs(x_centroid - x_samples[sample]) > 5 or abs(y_centroid - y_samples[sample]) > 5:
        #     continue
        # Calculate sample distance from pixel centroid
        centroid_distances[..., sample] = np.sqrt(
            (x_centroids - x_samples[sample]) ** 2 + (y_centroids - y_samples[sample]) ** 2)
    if channels > 1:
        power_grid = np.zeros(img_resolution)
        for channel in range(channels):
            pg = power[channel] * np.exp(-decay[channel] * centroid_distances)
            power_grid[..., channel] = np.sum(pg, axis=2)
    else:
        pg = power * np.exp(-decay * centroid_distances)
        power_grid = np.sum(pg, axis=2)

    # Prepare for grayscale image
    # Normalize Grid Array to 255 (8-bit pixel value)
    normalized_grid = (power_grid / np.max(power_grid)) * 255
    # Quantize grid to integers
    normalized_grid = np.floor(normalized_grid)
    # Copy result to uint8 array for writing grayscale image
    img_grid = np.empty(img_resolution, dtype='uint8')
    np.copyto(img_grid, normalized_grid, casting='unsafe')
    # Generate grayscale image from grid array
    if channels == 1:
        img = Image.fromarray(img_grid, mode='L')
    elif channels == 3:
        img = Image.fromarray(img_grid, mode='RGB')
    # Show Image
    # img.show()
    # Permanently Save Image
    img.save(filename)
