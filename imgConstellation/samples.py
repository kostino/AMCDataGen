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
        # Declare moments array - run calculateMoments to fill
        self.amp_moments = None
        self.phase_moments = None
        self.signal_moments = None
        # Declare moments array - run calculateMoments to fill
        self.amp_cumulants = None
        self.phase_cumulants = None
        self.signal_cumulants = None

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

    def calculateMoment(self, p, q):
        """
        Return moment M_p,q
        :param p: Parameter p
        :param q: Parameter q
        :return: M_pq
        """
        conj = np.conjugate(self.samples)
        return np.mean(np.power(self.samples, p-q) * np.power(conj, q))

    def calculateMoments(self):
        # # Amplitude & Phase
        # amp = np.abs(self.samples)
        # phase = np.angle(self.samples)
        #
        # self.amp_moments = {"M{}0".format(i): stats.moment(amp, i) for i in range(start=1, stop=7, step=1)}
        # self.phase_moments = {"M{}0".format(i): stats.moment(phase, i) for i in range(start=1, stop=7, step=1)}
        self.signal_moments = {"M{}{}".format(p,q): self.calculateMoment(p, q) for p in range(2, 7)
                               for q in range(0, 4) if q <= p}
        return self

    def calculateCumulants(self):
        # Call moments calculation
        self.calculateMoments()

        # Calculate Signal Cumulants
        self.signal_cumulants = {}
        self.signal_cumulants["C20"] = self.signal_moments["M20"]
        self.signal_cumulants["C21"] = self.signal_moments["M21"]
        self.signal_cumulants["C40"] = self.signal_moments["M40"] - 3 * np.power(self.signal_moments["M20"], 2)
        self.signal_cumulants["C41"] = self.signal_moments["M41"] \
                                       - 3 * self.signal_moments["M20"] * self.signal_moments["M21"]
        self.signal_cumulants["C42"] = self.signal_moments["M42"] - np.power(np.abs(self.signal_moments["M20"]), 2) \
                                       - 2 * np.power(self.signal_moments["M21"], 2)
        self.signal_cumulants["C60"] = self.signal_moments["M60"] - 15 * self.signal_moments["M20"] * self.signal_moments["M40"] \
                                       + 3 * np.power(self.signal_moments["M20"], 3)
        self.signal_cumulants["C61"] = self.signal_moments["M61"] - 5 * self.signal_moments["M21"] * self.signal_moments["M40"] \
                                       - 10 * self.signal_moments["M20"] * self.signal_moments["M41"] \
                                       + 30 * np.power(self.signal_moments["M20"], 2) * self.signal_moments["M21"]
        self.signal_cumulants["C62"] = self.signal_moments["M62"] - 6 * self.signal_moments["M20"] * self.signal_moments["M42"] \
                                       - 8 * self.signal_moments["M21"] * self.signal_moments["M41"] \
                                       - self.signal_moments["M22"] * self.signal_moments["M40"] \
                                       + 6 * np.power(self.signal_moments["M20"], 2) * self.signal_moments["M22"] \
                                       + 24 * np.power(self.signal_moments["M21"], 2) * self.signal_moments["M20"]
        self.signal_cumulants["C63"] = self.signal_moments["M63"] - 9 * self.signal_moments["M21"] * self.signal_moments["M42"] \
                                       + 12 * np.power(self.signal_moments["M21"], 3) \
                                       - 3 * self.signal_moments["M20"] * self.signal_moments["M43"] \
                                       - 3 * self.signal_moments["M22"] * self.signal_moments["M41"]  \
                                       + 18 * self.signal_moments["M20"] * self.signal_moments["M21"] * self.signal_moments["M22"]

        return self


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

    """ Plotting functions """
    def plot(self):
        """
        Plots the constellation
        :return:
        """
        plt.figure(figsize=(5, 5))
        plt.scatter(self.samples.real, self.samples.imag)
        plt.show()

    """ Data I/O functions """
    def saveSamples(self, filename):
        """
        Saves the raw I/Q samples to a binary file
        :param filename: File name to save the samples
        :return:
        """
        self.samples.tofile(filename)
        return self

    def loadSamples(self, filename):
        """
        Loads raw I/Q samples from a binary file
        :param filename: File name to load the samples
        :return:
        """
        self.samples = np.fromfile(filename, np.complex128)
        return self

    def saveCumulants(self, filename):
        """
        Saves the cumulants to a binary file
        :param filename: File name to save the samples
        :return:
        """
        cum_array = np.array([item for key, item in self.signal_cumulants])
        cum_array.tofile(filename)
        return self

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

        if decay is None or power is None:
            decay = (0.4, 0.3, 0.2)
            power = (100, 80, 60)

        imgGen.enhancedImgGen(self.samples, irange, qrange, img_resolution, filename, 3, power, decay, global_norm)

    def enhancedRGBCUDABATCH(self, img_resolution, filename, bounds=None, decay=None, power=None, n_images=1, global_norm=False):
        if bounds is not None:
            (irange, qrange) = bounds
        else:
            irange = self.irange
            qrange = self.qrange

        if decay is None or power is None:
            decay = (0.4, 0.3, 0.2)
            power = (100, 80, 60)

        imgGen.enhancedImgGenCUDABATCH(self.samples, irange, qrange, img_resolution, filename, 3, power, decay, n_images, global_norm)
