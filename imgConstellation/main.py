from utilities import IQplot
from constellations import *

from PIL import Image

# === Parameters ===

# Signal to Noise Ratio (SNR), aka Bit Energy (Eb) to Noise Power (N0) in Decibel
EbN0_db = 10


sixteenPSK = PSK("16-PSK", 16, 8, 45)
sixteenQAM = QAM("16-QAM", 16, 8, 45)
threeAPSK = APSK("SimpleAPSK", rings=3, symbols_num=[4, 8, 12], radii=[2, 4, 8], angle_offsets=[0, 45, 0])
print(sixteenPSK.symbols)
sixteenPSK.plot()
print(sixteenQAM.symbols)
sixteenQAM.plot()
print(threeAPSK.symbols)
threeAPSK.plot()

samplesAPSK = threeAPSK.sampleGenerator(15)
IQplot(samplesAPSK)
