from utilities import IQplot
from constellations import *
from imgGen import *
import time
import os
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

samplesAPSK = threeAPSK.sampleGenerator(1000, SNR=10).samples
IQplot(samplesAPSK)

# Generate Grayscale, Enhanced Grayscale and RGB images from a 3 ring APSK constellation
grayscaleImgGen(samplesAPSK, (-7, 7), (-7, 7), (200, 200), "constellation_gray.png")
enhancedImgGen(samplesAPSK, (-7, 7), (-7, 7), (200, 200), "constellation_enGray.png", 1, 50, 1)
enhancedImgGen(samplesAPSK, (-7, 7), (-7, 7), (200, 200), "constellation_rgb.png", 3, (100, 80, 60), (0.4, 0.3, 0.2))

# Calculate advised bounds and generate image using them
x_min, x_max, y_min, y_max = threeAPSK.bounds(SNR=10, stds_num=2, padding=5)
start_time = time.time()
enhancedImgGen(samplesAPSK, (x_min, x_max), (y_min, y_max), (224, 224), "constellation_rgb_bounds.png", 3,
               (2000, 2000, 2000), (0.4, 0.3, 0.2))
print(time.time() - start_time)

# Dataset 1 Generation
QPSK = PSK("QPSK", 4, 1, 0)
eight_PSK = PSK("8PSK", 8, 1, 0)
sixteen_QAM = QAM("16QAM", 16, 1, 0)
sixtyfour_QAM = QAM("64QAM", 64, 1, 0)
four_PAM = PAM("4PAM", 4, 1, 0)
sixteen_PAM = PAM("16PAM", 16, 1, 0)
sixteen_APSK = APSK("16APSK", 2, (8, 8), (0.8, 1.2), (0, 0))
sixtyfour_APSK = APSK("64APSK", 4, (8, 16, 20, 20), (0.3, 0.6, 0.9, 1.2), (0, 0, 0, 0))

batches = 15000
img_resolution = (224, 224)

# Iterate over SNRs
for snr in (0, 5, 10, 15):
    # Iterate over Modulation Schemes
    for modulation in (QPSK, eight_PSK, sixteen_QAM, sixtyfour_QAM, four_PAM, sixteen_PAM, sixteen_APSK, sixtyfour_APSK):
        print("Starting modulation {} for {}dB".format(modulation.name, snr))
        os.makedirs("data/{}_db/{}/".format(snr, modulation.name))
        for batch in range(batches):
            # print(batch)
            # Generate samples
            samples = modulation.sampleGenerator(samples_num=1000, SNR=snr)
            samples.enhancedRGB(img_resolution=img_resolution,
                                filename="data/{}_db/{}/{}.png".format(snr, modulation.name, batch))
        confirmation = input("{}db {} is done, continue?".format(snr, modulation))



# New simplified way of creating images

sixteenPSK.sampleGenerator(1000, SNR=10).enhancedRGB((200, 200), "psk_rgb.png")
