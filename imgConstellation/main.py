import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# === Parameters ===

# Signal to Noise Ratio (SNR), aka Bit Energy (Eb) to Noise Power (N0) in Decibel
EbN0_db = 10


# === Constellations ===

class Constellation:
    """Modulation Scheme Base Class"""

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

    # Plot constellation
    def plot(self):
        plt.figure(figsize=(5, 5))
        plt.scatter(self.symbols.real, self.symbols.imag)
        plt.show()


class PSK(Constellation):
    """PSK Modulation Scheme Class"""

    def __init__(self, name, symbols_num, symbol_power, angle_offset):
        # Call parent constructor
        super().__init__(name, symbols_num)
        # Symbol Power
        self.symbol_power = symbol_power
        # Constellation offset
        self.angle_offset = angle_offset
        # Symbols angles
        self.angles = np.arange(symbols_num) * 2 * np.pi / symbols_num + angle_offset
        # Symbols
        self.symbols = symbol_power * np.exp(-1j * self.angles)


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
        self.symbols = self.symbols * (np.cos(angle_offset)+1j*np.sin(angle_offset))


sixteenPSK = PSK("16-PSK", 16, 8, 45)
sixteenQAM = QAM("16-QAM", 16, 8, 45)
print(sixteenPSK.symbols)
sixteenPSK.plot()
print(sixteenQAM.symbols)
sixteenQAM.plot()
