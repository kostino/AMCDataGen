# import os
# os.environ['NUMEXPR_MAX_THREADS'] = '16'
# os.environ['NUMEXPR_NUM_THREADS'] = '8'
import numpy as np
from PIL import Image
import numexpr as ne
import cupy as cp


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
def enhancedImgGen(symbols, i_range, q_range, img_resolution, filename, channels, power, decay, global_norm=False):
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
    :param global_norm: Whether to normalize the image pixels on a global-across all channels or on a per-channel basis
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

    # Calculate pixel centroids in continuous x,y plane
    x_centroids = np.arange(start=0.5, stop=img_resolution[0], step=1, dtype='float32').reshape(
        (1, img_resolution[0], 1))
    y_centroids = np.arange(start=0.5, stop=img_resolution[1], step=1, dtype='float32').reshape(
        (1, 1, img_resolution[1]))

    x_samples = np.array(x_samples).reshape((samples_num, 1, 1))
    y_samples = np.array(y_samples).reshape((samples_num, 1, 1))

    centroid_distances = ne.evaluate("sqrt((x_centroids - x_samples) ** 2 + (y_centroids - y_samples) ** 2)")
    if channels > 1:
        power_grid = np.zeros(img_resolution)
        for channel in range(channels):
            d = decay[channel]
            p = power[channel]
            pg = ne.evaluate('p / exp(d * centroid_distances)')
            power_grid[..., channel] = np.sum(pg, axis=0)
    else:
        pg = ne.evaluate('power / exp(decay * centroid_distances)')
        power_grid = np.sum(pg, axis=0)

    # Prepare for grayscale image
    # Normalize Grid Array to 255 (8-bit pixel value)
    if not global_norm:
        # Normalize on a global basis
        normalized_grid = (power_grid / np.max(power_grid, axis=(0, 1)).reshape((1, 1, channels))) * 255
    else:
        # Normalize on a per channel basis
        normalized_grid = (power_grid / np.max(power_grid)) * 255
    # Quantize grid to integers
    normalized_grid = np.floor(normalized_grid)
    # Copy result to uint8 array for writing grayscale image
    img_grid = normalized_grid.astype('uint8', casting='unsafe')
    # Generate grayscale image from grid array
    if channels == 1:
        img = Image.fromarray(img_grid[0], mode='L')
    elif channels == 3:
        img = Image.fromarray(img_grid, mode='RGB')
    # Show Image
    # img.show()
    # Permanently Save Image
    try:
        img.save(filename)
    except NameError:
        print("Only single and 3-channel images supported")


# === Enhanced Grayscale and RGB Image Generation - Section III-C&D ===
def enhancedImgGenCUDABATCH(symbols, i_range, q_range, img_resolution, filename, channels, power, decay, n_images, global_norm=False):
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
    :param global_norm: Whether to normalize the image pixels on a global-across all channels or on a per-channel basis
    :return:
    """
    filename = filename.replace(".png","",1)
    # Transform I/Q samples to XY plane
    x_samples, y_samples = IQtoXY(symbols, i_range, q_range, img_resolution)

    # Number of final samples
    samples_num = len(x_samples)
    samples_per_image = samples_num // n_images
    samples_to_keep = n_images * samples_per_image

    # Add channel number to dimensions if dimensions > 1
    if channels > 1:
        img_resolution = (img_resolution[0], img_resolution[1], channels)

    # Numpy array representing the 'power' of each pixel value as influenced by each sample

    # Calculate pixel centroids in continuous x,y plane
    x_centroids = cp.arange(start=0.5, stop=img_resolution[0], step=1, dtype='float32').reshape(
        (1, img_resolution[0], 1, 1, 1))
    y_centroids = cp.arange(start=0.5, stop=img_resolution[1], step=1, dtype='float32').reshape(
        (1, 1, img_resolution[1], 1, 1))

    x_samples = cp.array(x_samples[:samples_to_keep], dtype='float32').reshape((n_images, 1, 1, 1, samples_per_image))
    y_samples = cp.array(y_samples[:samples_to_keep], dtype='float32').reshape((n_images, 1, 1, 1, samples_per_image))

    decay = cp.array(decay, dtype='float32').reshape((1, 1, 1, channels, 1))
    power = cp.array(power, dtype='float32').reshape((1, 1, 1, channels, 1))

    pg = cp.ElementwiseKernel('float32 x, float32 x_c, float32 y, float32 y_c, float32 decay, float32 power',
                              'float32 z',
                              'z = power / exp(decay * sqrt((x - x_c) * (x - x_c) + (y - y_c) * (y - y_c)))',
                              'dist')

    powergrid = cp.empty((n_images, img_resolution[0], img_resolution[1], channels, samples_per_image), dtype='float32')
    pg(x_samples, x_centroids, y_samples, y_centroids, decay, power, powergrid)
    power_grid = cp.asnumpy(cp.sum(powergrid, axis=4))

    # Normalize Grid Array to 255 (8-bit pixel value)
    if not global_norm:
        # Normalize on a global basis
        normalized_grid = (power_grid / np.max(power_grid, axis=(0, 1, 2)).reshape((1, 1, 1, channels))) * 255
    else:
        # Normalize on a per channel basis
        normalized_grid = (power_grid / np.max(power_grid)) * 255
    # Quantize grid to integers
    normalized_grid = np.floor(normalized_grid)
    # Copy result to uint8 array for writing grayscale image
    img_grid = normalized_grid.astype('uint8', casting='unsafe')
    # Generate grayscale image from grid array
    if channels == 1:
        img = Image.fromarray(img_grid, mode='L')
    elif channels == 3:
        for i in range(n_images):
            img = Image.fromarray(img_grid[i], mode='RGB')
            try:
                img.save('{}_{}.png'.format(filename, i))
            except NameError:
                print("Only single and 3-channel images supported")
    # Show Image
    # img.show()
    # Permanently Save Image
    # try:
    #     img.save(filename)
    # except NameError:
    #     print("Only single and 3-channel images supported")
