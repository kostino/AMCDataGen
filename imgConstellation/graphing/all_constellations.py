from constellations import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


QPSK = PSK("QPSK", 4, 1, 0)
eight_PSK = PSK("8PSK", 8, 1, 0)
sixteen_QAM = QAM("16QAM", 16, 1, 0)
sixtyfour_QAM = QAM("64QAM", 64, 1, 0)
four_PAM = PAM("4PAM", 4, 1, 0)
sixteen_PAM = PAM("16PAM", 16, 1, 0)
sixteen_APSK = APSK("16APSK", 2, (8, 8), (0.8, 1.2), (0, 0))
sixtyfour_APSK = APSK("64APSK", 4, (8, 16, 20, 20), (0.3, 0.6, 0.9, 1.2), (0, 0, 0, 0))

img_resolution = (224, 224)

# Plot examples of all the modulation schemes
# fig, ax = plt.subplots(3,3, figsize=(8, 8))
fig, ax = plt.subplots(2,4, figsize=(8, 4))
fig.tight_layout(h_pad=2)
# plt.suptitle("Simulations for various Modulations (30dB SNR AWGN)", fontsize=20)
for index, modulation in enumerate((QPSK, eight_PSK, sixteen_QAM, sixtyfour_QAM, four_PAM, sixteen_PAM, sixteen_APSK, sixtyfour_APSK)):
    # axi = plt.subplot(3,3, index+1)
    axi = plt.subplot(2,4, index+1)
    samples = modulation.sampleGenerator(samples_num=1000).awgn(SNR=17)
    samples.enhancedRGB(img_resolution, f'{modulation.name}.png', decay=(0.4, 0.2, 0.1))
    axi.set_title(modulation.name)
    # plt.xlim([-2, 2])
    # plt.ylim([-2, 2])
    # plt.scatter(samples.samples.real, samples.samples.imag)
    img = mpimg.imread(f'{modulation.name}.png')
    plt.imshow(img)
    plt.axis('off')
# plt.subplots_adjust(top=0.88)
plt.show()
