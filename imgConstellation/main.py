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


sixteenPSK = PSK("16-PSK", 16, 8, 45)
print(sixteenPSK.symbols)
sixteenPSK.plot()
