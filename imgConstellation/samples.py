import matplotlib.pyplot as plt
import numpy as np
import imgGen


class Samples:
    """
    Sample collection class
    ...

            Attributes:
            -----------
                modulation (string): Name of the modulation scheme
                samples_num (int): Number of samples
                sample (complex array): Samples on the I/Q plane

            Methods:
            -----------
                plot(): Plots the samples on the IQ plane
    """

    # Constructor
    def __init__(self, modulation, samples_num, symbols, symbol_power, SNR=None):
        # Modulation Scheme Name
        self.modulation = modulation
        # Number of Symbols
        self.samples_num = samples_num
        # Uninitialized array to place symbols
        self.symbols = symbols
        # Initialize the default random generator
        self.rng = np.random.default_rng()

        indexes = self.rng.integers(0, len(symbols), samples_num)
        samples = symbols[indexes]
        if SNR is not None:
            gamma = np.power(10, SNR / 10)
            N0 = symbol_power / gamma
            n = np.sqrt(N0 / 2) * (np.random.randn(samples_num) + 1j * np.random.randn(samples_num))  # AWGN
            samples = samples + n  # Apply AWGN to samples
        self.samples = samples
        x_max = np.max(np.abs(self.symbols.real))
        y_max = np.max(np.abs(self.symbols.imag))
        max_range = np.max(np.array((x_max, y_max)))

        # Calculate N0 and extend boundaries by stds_nums * np.sqrt(N0/2)
        # Setting strds_nums=2 should contain ~90% of samples from symbols near the edges
        if SNR is not None:
            gamma = np.power(10, SNR / 10)
            N0 = symbol_power / gamma
            max_range += 2 * np.sqrt(N0 / 2)

        # Add padding
        offset = (100 * 2 * max_range / (100 - 5) - (2 * max_range)) / 2
        scale = offset + max_range
        self.irange = (-scale, scale)
        self.qrange = (-scale, scale)

    def plot(self):
        '''

        Plots the constellation
        :return:
        '''
        plt.figure(figsize=(5, 5))
        plt.scatter(self.samples.real, self.samples.imag)
        plt.show()

    def grayscale(self, img_resolution, filename, bounds=None):
        if bounds is not None:
            (irange, qrange) = bounds
            imgGen.grayscaleImgGen(self.samples, irange, qrange, img_resolution, filename)
        else:
            imgGen.grayscaleImgGen(self.samples, self.irange, self.qrange, img_resolution, filename)

    def enhancedGrayscale(self, img_resolution, filename, bounds=None, decay=None, power=None):
        if bounds is not None:
            (irange, qrange) = bounds
        else:
            irange = self.irange
            qrange = self.qrange

        if decay is None or power is None:
            decay = 0.2
            power = 50

        imgGen.enhancedImgGen(self.samples, irange, qrange, img_resolution, filename, 1, power, decay)

    def enhancedRGB(self, img_resolution, filename, bounds=None, decay=None, power=None):
        if bounds is not None:
            (irange, qrange) = bounds
        else:
            irange = self.irange
            qrange = self.qrange

        if decay is None or power is None:
            decay = (0.4, 0.3, 0.2)
            power = (100, 80, 60)

        imgGen.enhancedImgGen(self.samples, irange, qrange, img_resolution, filename, 3, power, decay)

    def enhancedRGBCUDA(self, img_resolution, filename, bounds=None, decay=None, power=None):
        if bounds is not None:
            (irange, qrange) = bounds
        else:
            irange = self.irange
            qrange = self.qrange

        if decay is None or power is None:
            decay = (0.4, 0.3, 0.2)
            power = (100, 80, 60)

        imgGen.enhancedImgGenCUDA(self.samples, irange, qrange, img_resolution, filename, 3, power, decay)
