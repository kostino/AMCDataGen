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
# EbN0_db = 10


# fourPAM = PAM("4-PAM", 4, 8, 0)
# sixteenPSK = PSK("16-PSK", 16, 8, 45)
# sixteenQAM = QAM("16-QAM", 16, 8, 45)
# threeAPSK = APSK("SimpleAPSK", rings=3, symbols_num=[4, 8, 12], radii=[2, 4, 6], angle_offsets=[0, 45, 0])
# print(fourPAM.symbols)
# fourPAM.plot()
# print(sixteenPSK.symbols)
# sixteenPSK.plot()
# print(sixteenQAM.symbols)
# sixteenQAM.plot()
# print(threeAPSK.symbols)
# threeAPSK.plot()

# samplesAPSK = threeAPSK.sampleGenerator(1000).awgn(SNR=10)
# samplesAPSK.plot()

# # Generate Grayscale, Enhanced Grayscale and RGB images from a 3 ring APSK constellation
# grayscaleImgGen(samplesAPSK, (-7, 7), (-7, 7), (200, 200), "constellation_gray.png")
# enhancedImgGen(samplesAPSK, (-7, 7), (-7, 7), (200, 200), "constellation_enGray.png", 1, 50, 1, False)
# enhancedImgGen(samplesAPSK, (-7, 7), (-7, 7), (200, 200), "constellation_rgb.png", 3, (100, 80, 60), (0.4, 0.3, 0.2), False)

# # Calculate advised bounds and generate image using them
# x_min, x_max, y_min, y_max = threeAPSK.bounds(SNR=10, stds_num=2, padding=5)
# start_time = time.time()
# enhancedImgGen(samplesAPSK, (x_min, x_max), (y_min, y_max), (224, 224), "constellation_rgb_bounds.png", 3,
#                (2000, 2000, 2000), (0.4, 0.3, 0.2))
# print(time.time() - start_time)

# Dataset Generation
data_root = 'dataset5'
data_root_iq = 'dataset5_iq'
data_root_sig_cum = 'dataset5_cum'
QPSK = PSK("QPSK", 4, 1, 0)
eight_PSK = PSK("8PSK", 8, 1, 0)
sixteen_QAM = QAM("16QAM", 16, 1, 0)
sixtyfour_QAM = QAM("64QAM", 64, 1, 0)
four_PAM = PAM("4PAM", 4, 1, 0)
sixteen_PAM = PAM("16PAM", 16, 1, 0)
sixteen_APSK = APSK("16APSK", 2, (8, 8), (0.8, 1.2), (0, 0))
sixtyfour_APSK = APSK("64APSK", 4, (8, 16, 20, 20), (0.3, 0.6, 0.9, 1.2), (0, 0, 0, 0))

size = 15000
img_resolution = (224, 224)
batch_size = 5
iter_size = size // batch_size

# Iterate over Modulation Schemes
for modulation in (QPSK, eight_PSK, sixteen_QAM, sixtyfour_QAM, four_PAM, sixteen_PAM, sixteen_APSK, sixtyfour_APSK):
    os.makedirs("{}/{}".format(data_root, modulation.name))
    os.makedirs("{}/{}".format(data_root_iq, modulation.name))
    os.makedirs("{}/{}".format(data_root_sig_cum, modulation.name))
    # Iterate over SNRs
    for snr in (0, 5, 10, 15):
        print("Starting {}dB for modulation {}".format(snr, modulation.name))
        if snr not in os.listdir("{}/{}".format(data_root, modulation.name)):
            os.makedirs("{}/{}/{}_db/".format(data_root, modulation.name, snr))
            os.makedirs("{}/{}/{}_db/".format(data_root_iq, modulation.name, snr))
            os.makedirs("{}/{}/{}_db/".format(data_root_sig_cum, modulation.name, snr))
            for iteration in range(iter_size):
                # Generate samples and apply AWGN
                samples = modulation.sampleGenerator(samples_num=1000*batch_size).awgn(SNR=snr)
                # Generate image
                samples.enhancedRGBCUDABATCH(img_resolution=img_resolution,
                                     filename="{}/{}/{}_db/{}.png".format(data_root, modulation.name, snr, iteration),
                                     n_images=batch_size, decay=(0.4,0.2,0.1))
                # Save raw I/Q samples to binary file
                samples.saveSamplesBATCH(filename="{}/{}/{}_db/{}.iq".format(data_root_iq, modulation.name, snr, iteration), batch_size=batch_size)
                # Calculate and save cumulants
                samples.calculateCumulantsBATCH(batch_size=batch_size).saveCumulantsBATCH(filename="{}/{}/{}_db/{}.cum".format(data_root_sig_cum, modulation.name, snr, iteration))
        else:
            print("Skipping {}dB for modulation {}. Already completed".format(modulation.name, snr))
