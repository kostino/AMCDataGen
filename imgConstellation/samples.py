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
                samples (complex array): Samples on the I/Q plane
                symbol_power (float): Average symbol power from constellation

            Methods:
            -----------
                plot(): Plots the samples on the IQ plane
    """

    # Constructor
    def __init__(self, modulation, samples_num, symbols, symbol_power):
        # Modulation Scheme Name
        self.modulation = modulation
        # Number of Symbols
        self.samples_num = samples_num
        # Uninitialized array to place symbols
        self.symbols = symbols
        # Initialize the default random generator
        self.rng = np.random.default_rng()
        # Save average symbol power
        self.symbol_power = symbol_power

        # Generate random indices and use them to generate a series of symbols samples
        indexes = self.rng.integers(0, len(symbols), samples_num)
        samples = symbols[indexes]
        self.samples = samples

        # Calculate maximum x or y distance of samples and set max_range variable
        x_max = np.max(np.abs(self.symbols.real))
        y_max = np.max(np.abs(self.symbols.imag))
        self.max_range = np.max(np.array((x_max, y_max)))

        # Add padding
        offset = (100 * 2 * self.max_range / (100 - 5) - (2 * self.max_range)) / 2
        scale = offset + self.max_range
        self.irange = (-scale, scale)
        self.qrange = (-scale, scale)

    """ Channel Impairments functions """
    def awgn(self, SNR):
        """
        Applies Additive White Gaussian Noise to samples for a given Signal to Noise Ratio (SNR)
        :param SNR: Signal to Noise Ratio in dB to apply Additive White Gaussian Noise (optional)
        :return:
        """
        # Signal to Noise ratio in linear scale (non-dB)
        gamma = np.power(10, SNR / 10)
        # Noise power
        N0 = self.symbol_power / gamma
        # Generate AWGN
        n = np.sqrt(N0 / 2) * (np.random.randn(self.samples_num) + 1j * np.random.randn(self.samples_num))
        # Apply AWGN to samples
        self.samples = self.samples + n
        # Extend boundaries by stds_nums * np.sqrt(N0/2)
        # Setting strds_nums=2 should contain ~90% of samples from symbols near the edges
        self.max_range += 2 * np.sqrt(N0 / 2)
        # Update padding
        offset = (100 * 2 * self.max_range / (100 - 5) - (2 * self.max_range)) / 2
        scale = offset + self.max_range
        self.irange = (-scale, scale)
        self.qrange = (-scale, scale)

        return self

    def freqOffset(self, sample_freq=None, offset_freq=None, degrees=None):
        """
        Simulates a frequency offset between transmitter and receiver. Enter either sample_freq with offset_freq or just
        degrees.
        :param sample_freq: Base transmitter and receiver frequency in Hz
        :param offset_freq: Frequency offset between transmitter and receiver in Hz
        :param degrees: Degrees of rotation / phase offset to apply
        :return:
        """

        # Using sampling frequency and offset frequency
        if (sample_freq is not None) and (offset_freq is not None):
            self.samples = self.samples * np.exp(1j*2*np.pi*(offset_freq/sample_freq))
        # Using degrees
        elif degrees is not None:
            self.samples = self.samples * np.exp(1j*2*np.pi*degrees)


    """ Plotting functions """
    def plot(self):
        """
        Plots the constellation
        :return:
        """
        plt.figure(figsize=(5, 5))
        plt.scatter(self.samples.real, self.samples.imag)
        plt.show()

    """ Image generation functions """
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

        imgGen.enhancedImgGen(self.samples, irange, qrange, img_resolution, filename, 1, power, decay, False)

    def enhancedRGB(self, img_resolution, filename, bounds=None, decay=None, power=None, global_norm=False):
        if bounds is not None:
            (irange, qrange) = bounds
        else:
            irange = self.irange
            qrange = self.qrange

        if decay is None:
            decay = (0.4, 0.3, 0.2)
        power = (100, 80, 60)

        imgGen.enhancedImgGen(self.samples, irange, qrange, img_resolution, filename, 3, power, decay, global_norm)

    def enhancedRGBCUDABATCH(self, img_resolution, filename, bounds=None, decay=None, power=None, n_images=1, global_norm=False):
        if bounds is not None:
            (irange, qrange) = bounds
        else:
            irange = self.irange
            qrange = self.qrange

        if decay is None:
            decay = (0.4, 0.3, 0.2)
        power = (100, 80, 60)

        imgGen.enhancedImgGenCUDABATCH(self.samples, irange, qrange, img_resolution, filename, 3, power, decay, n_images, global_norm)
