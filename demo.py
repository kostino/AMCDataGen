from imgConstellation.constellations import PSK, QAM, PAM, APSK
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Available modulations
QPSK = PSK("QPSK", 4, 1, 0)
eight_PSK = PSK("8PSK", 8, 1, 0)
sixteen_QAM = QAM("16QAM", 16, 1, 0)
sixtyfour_QAM = QAM("64QAM", 64, 1, 0)
four_PAM = PAM("4PAM", 4, 1, 0)
sixteen_PAM = PAM("16PAM", 16, 1, 0)
sixteen_APSK = APSK("16APSK", 2, (8, 8), (0.8, 1.2), (0, 0))
sixtyfour_APSK = APSK("64APSK", 4, (8, 16, 20, 20), (0.3, 0.6, 0.9, 1.2), (0, 0, 0, 0))


# Config
img_resolution = (224, 224)
decay_f = (0.4, 0.2, 0.1)
snr = 17
samples_num = 1000
save_folder = './demo'

fig, ax = plt.subplots(2, 4, figsize=(8, 4))
for index, modulation in enumerate((QPSK, eight_PSK, sixteen_QAM, sixtyfour_QAM, four_PAM, sixteen_PAM, sixteen_APSK, sixtyfour_APSK)):
    axi = plt.subplot(2, 4, index+1)
    samples = modulation.sampleGenerator(samples_num=samples_num).awgn(SNR=snr)
    samples.enhancedRGB(img_resolution, f'{save_folder}/{modulation.name}.png', decay=decay_f)
    axi.set_title(modulation.name)
    img = mpimg.imread(f'{save_folder}/{modulation.name}.png')
    plt.imshow(img)
    plt.axis('off')
plt.show()
