import numpy as np
import matplotlib.pyplot as plt
from samples import Samples


# === Constellations ===

class Constellation:
    """
    Modulation Scheme Base Class
    ...

            Attributes:
            -----------
                name (string): Name of the modulation scheme
                symbols_num (int): Number of symbols for the modulation scheme
                symbol_bits (int): Bits per symbol
                symbols (complex array): Symbols on the I/Q plane

            Methods:
            -----------
                plot(): Plots the constellation from symbols

    """

    # Constructor
    def __init__(self, name, symbols_num, symbol_power):
        '''
        Constructor function for Constellation

        :param name: String to identify constellation
        :param symbols_num: Number of symbols
        :param symbol_power: Symbol power
        '''
        # Modulation Scheme Name
        self.name = name
        # Number of Symbols
        self.symbols_num = symbols_num
        # Average Symbol Power
        self.symbol_power = symbol_power
        # Bits per symbol
        self.symbol_bits = np.log2(symbols_num)
        # Uninitialized array to place symbols
        self.symbols = np.empty(symbols_num)
        # Initialize the default random generator
        self.rng = np.random.default_rng()

    # Plot constellation
    def plot(self):
        '''
        Plots the constellation
        :return:
        '''
        plt.figure(figsize=(5, 5))
        plt.scatter(self.symbols.real, self.symbols.imag)
        plt.show()

    # Generate Samples and applies AWGN. SNR is in db
    def sampleGenerator(self, samples_num):
        '''
        Generates random samples from the constellation by sampling a discrete uniform distribution
        :param samples_num: Number of samples to be generated
        :return: Array of complex constellation samples
        '''
        samples = Samples(self.name, samples_num, self.symbols, self.symbol_power)
        return samples

    # Calculates advised bounds in the I/Q plane
    def bounds(self, SNR=None, stds_num=0, padding=0):
        '''
        Calculates advised bounds in the I/Q plane for both I and Q axis
        :param SNR: Signal to Noise Ratio in dB to account for Additive White Gaussian Noise (AWGN)
        :param stds_num: Padding factor based on AWGN's power / Standard Deviation. E.g: 2 covers ~90% of edge samples,
        3 covers ~99%
        :param padding: Image padding on the edges in percent (%) based on each axis bandwidth.
        :return: Bound indicators for I and Q axes: I_minimum, I_maximum, Q_minimum, Q_maximum
        '''
        # Calculate max real and imaginary components
        x_max = np.max(np.abs(self.symbols.real))
        y_max = np.max(np.abs(self.symbols.imag))
        max_range = np.max(np.array((x_max, y_max)))

        # Calculate N0 and extend boundaries by stds_nums * np.sqrt(N0/2)
        # Setting strds_nums=2 should contain ~90% of samples from symbols near the edges
        if SNR is not None:
            gamma = np.power(10, SNR / 10)
            N0 = self.symbol_power / gamma
            max_range += stds_num * np.sqrt(N0 / 2)

        # Add padding
        offset = (100 * 2 * max_range / (100 - padding) - (2 * max_range)) / 2
        scale = offset + max_range
        irange = (-scale, scale)
        qrange = (-scale, scale)

        return -scale, scale, -scale, scale


class PSK(Constellation):
    """
    PSK Modulation Scheme Class
    ...
            Inherits:
            -----------
                Constellation

            Attributes:
            -----------
                name (string): Name of the modulation scheme
                symbols_num (int): Number of symbols for the modulation scheme
                symbol_bits (int): Bits per symbol
                symbols (complex array): Symbols on the I/Q plane
                radius (int): PSK constellation radius
                angle_offset (int): Rotation offset for the constellation in degrees

            Methods:
            -----------
                plot(): Plots the constellation from symbols

    """

    def __init__(self, name, symbols_num, radius, angle_offset):
        # Find symbol power first
        symbol_power = radius ** 2
        # Call parent constructor
        super().__init__(name, symbols_num, symbol_power)
        # Symbol Ring Radius
        self.radius = radius
        # Constellation offset
        self.angle_offset = angle_offset
        # Symbols angles
        self.angles = np.arange(symbols_num) * 2 * np.pi / symbols_num + angle_offset
        # Symbols
        self.symbols = radius * np.exp(-1j * self.angles)


class APSK(Constellation):
    """
    APSK Modulation Scheme Class
    ...
            Inherits:
            -----------
                Constellation

            Attributes:
            -----------
                name (string): Name of the modulation scheme
                rings (int): Number of rings for the APSK constellation
                symbols_num (int array): Number of symbols on each APSK ring
                symbol_bits (int): Bits per symbol
                symbols (complex array): Symbols on the I/Q plane
                radii (int array): Radius for each APSK ring
                angle_offsets (int array): Rotation offsets in degrees for each APSK constellation ring

            Methods:
            -----------
                plot(): Plots the constellation from symbols

    """

    def __init__(self, name, rings, symbols_num, radii, angle_offsets):
        # Call parent constructor
        n = sum(symbols_num)
        p = np.array(symbols_num) / n
        symbol_power = np.sum(np.power(np.array(radii), 2) * p)
        super().__init__(name, np.sum(symbols_num), symbol_power)
        # Number of rings
        self.rings = rings
        # Radii for symbol rings
        self.radii = radii
        # Angle offsets for symbol rings
        self.angle_offsets = angle_offsets
        # Symbol angles for each ring. It's a list of numpy arrays of variable length
        self.angles = []
        for ring in range(rings):
            self.angles.append(np.arange(symbols_num[ring]) * 2 * np.pi / symbols_num[ring] + angle_offsets[ring])
        # Symbols
        angles_concatenated = np.concatenate(self.angles)
        radii_expanded = np.concatenate([radii[ring] * np.ones(symbols_num[ring]) for ring in range(rings)])
        self.symbols = radii_expanded * np.exp(-1j * angles_concatenated)


class QAM(Constellation):
    """QAM Modulation Scheme Class"""

    def __init__(self, name, symbols_num, symbol_power, angle_offset):
        # Call parent constructor
        super().__init__(name, symbols_num, symbol_power)
        m = np.sqrt(symbols_num).astype(int)
        sum_term = m ** 2 * (m ** 2 - 1) / 6  # CORRECT EXPRESSION
        d_min = np.sqrt(symbols_num * symbol_power / sum_term)  # calculate dmin from avg symbol power
        # Constellation offset
        self.angle_offset = angle_offset
        n = np.arange(0, symbols_num)  # Sequential address from 0 to M-1 (1xM dimension)
        a = np.asarray([x ^ (x >> 1) for x in n])  # convert linear addresses to Gray code
        D = np.sqrt(symbols_num).astype(int)  # Dimension of K-Map - N x N matrix
        a = np.reshape(a, (D, D))  # NxN gray coded matrix
        oddRows = np.arange(start=1, stop=D, step=2)  # identify alternate rows
        nGray = np.reshape(a, symbols_num)  # reshape to 1xM - Gray code walk on KMap
        inputGray = n
        (x, y) = np.divmod(inputGray, D)  # element-wise quotient and remainder
        Ax = d_min * x + d_min / 2 * (1 - D)  # PAM Amplitudes 2d+1-D - real axis
        Ay = d_min * y + d_min / 2 * (1 - D)  # PAM Amplitudes 2d+1-D - imag axis
        self.symbols = Ax + 1j * Ay
        # apply angle offset
        self.symbols = self.symbols * (np.cos(angle_offset) + 1j * np.sin(angle_offset))


class PAM(Constellation):
    """PAM Modulation Scheme Class"""

    def __init__(self, name, symbols_num, symbol_power, angle_offset):
        # Call parent constructor
        super().__init__(name, symbols_num, symbol_power)
        # Constellation offset
        self.angle_offset = angle_offset
        egPAM = 3 * symbol_power / (symbols_num ** 2 - 1)
        d_min = 2 * np.sqrt(egPAM)
        n = np.arange(0, symbols_num)  # Sequential address from 0 to M-1 (1xM dimension)
        D = symbols_num
        Ax = d_min * n + d_min / 2 * (1 - D)  # PAM Amplitudes 2d+1-D - real axis
        Ay = 0  # PAM Amplitudes 0 - imag axis
        # apply angle offset
        self.symbols = (Ax + 1j * Ay) * (np.cos(angle_offset) + 1j * np.sin(angle_offset))
