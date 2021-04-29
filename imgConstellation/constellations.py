import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, name, symbols_num):
        # Modulation Scheme Name
        self.name = name
        # Number of Symbols
        self.symbols_num = symbols_num
        # Bits per symbol
        self.symbol_bits = np.log2(symbols_num)
        # Uninitialized array to place symbols
        self.symbols = np.empty(symbols_num)
        # Initialize the default random generator
        self.rng = np.random.default_rng()

    # Plot constellation
    def plot(self):
        plt.figure(figsize=(5, 5))
        plt.scatter(self.symbols.real, self.symbols.imag)
        plt.show()

    # Generate Samples
    def sampleGenerator(self, samples_num):
        indexes = self.rng.integers(0, self.symbols_num, samples_num)
        samples = self.symbols[indexes]
        return samples


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
        # Call parent constructor
        super().__init__(name, symbols_num)
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
        super().__init__(name, np.sum(symbols_num))
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
        super().__init__(name, symbols_num)
        # Symbol Power
        self.symbol_power = symbol_power
        # Constellation offset
        self.angle_offset = angle_offset
        # # Symbols angles
        # self.angles = np.arange(symbols_num) * 2 * np.pi / symbols_num + angle_offset
        # # Symbols
        # self.symbols = symbol_power * np.exp(-1j * self.angles)
        n = np.arange(0, symbols_num)  # Sequential address from 0 to M-1 (1xM dimension)
        a = np.asarray([x ^ (x >> 1) for x in n])  # convert linear addresses to Gray code
        D = np.sqrt(symbols_num).astype(int)  # Dimension of K-Map - N x N matrix
        a = np.reshape(a, (D, D))  # NxN gray coded matrix
        oddRows = np.arange(start=1, stop=D, step=2)  # identify alternate rows
        nGray = np.reshape(a, (symbols_num))  # reshape to 1xM - Gray code walk on KMap
        inputGray = n
        (x, y) = np.divmod(inputGray, D)  # element-wise quotient and remainder
        Ax = 2 * x + 1 - D  # PAM Amplitudes 2d+1-D - real axis
        Ay = 2 * y + 1 - D  # PAM Amplitudes 2d+1-D - imag axis
        self.symbols = Ax + 1j * Ay
        # apply angle offset
        self.symbols = self.symbols * (np.cos(angle_offset) + 1j * np.sin(angle_offset))

