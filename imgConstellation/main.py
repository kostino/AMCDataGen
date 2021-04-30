from utilities import IQplot
from constellations import *
from imgGen import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# === Parameters ===

# Signal to Noise Ratio (SNR), aka Bit Energy (Eb) to Noise Power (N0) in Decibel
EbN0_db = 10


fourPAM = PAM("4-PAM", 4, 8, 0)
sixteenPSK = PSK("16-PSK", 16, 8, 45)
sixteenQAM = QAM("16-QAM", 16, 8, 45)
threeAPSK = APSK("SimpleAPSK", rings=3, symbols_num=[4, 8, 12], radii=[2, 4, 6], angle_offsets=[0, 45, 0])
print(fourPAM.symbols)
fourPAM.plot()
print(sixteenPSK.symbols)
sixteenPSK.plot()
print(sixteenQAM.symbols)
sixteenQAM.plot()
print(threeAPSK.symbols)
threeAPSK.plot()

samplesAPSK = threeAPSK.sampleGenerator(15)
IQplot(samplesAPSK)

# Generate Grayscale and Enhanced Grayscale images from a 3 ring APSK constellation
grayscale(threeAPSK.symbols, (-7, 7), (-7, 7), (200, 200), "constellation.png")
enhancedGrayscale(threeAPSK.symbols, (-7, 7), (-7, 7), (200, 200), "constellation_en.png", 50, 1)
