from constellations import *
import matplotlib.pyplot as plt

four_PAM = PAM("4PAM", 4, 1, 0)
eight_PSK = PSK("8PSK", 8, 1, 0)
sixteen_QAM = QAM("16QAM", 16, 1, 0)
sixteen_APSK = APSK("16APSK", 2, (8, 8), (0.8, 1.2), (0, 0))


for index, mod in enumerate((four_PAM, eight_PSK, sixteen_QAM, sixteen_APSK)):
    pure = mod.sampleGenerator(1000).awgn(100)
    samples = mod.sampleGenerator(1000).awgn(20)

    fig = plt.figure(index+1)


    ax1 = fig.add_subplot(121)
    plt.xlim(samples.irange)
    plt.ylim(samples.irange)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(pure.samples.real, pure.samples.imag)
    plt.xticks([])
    plt.yticks([])
    ax1.title.set_text("Constellation")
    ax1.autoscale_view("tight")

    ax2 = fig.add_subplot(122)
    plt.xlim(samples.irange)
    plt.ylim(samples.irange)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(samples.samples.real, samples.samples.imag, s=1)
    plt.xticks([])
    plt.yticks([])
    ax2.title.set_text("Samples with AWGN, SNR=20")
    ax2.autoscale_view("tight")

    plt.show()

    fig.savefig(f"{mod.name}.png")
    fig.savefig(f"{mod.name}.eps")

