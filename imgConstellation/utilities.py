import matplotlib.pyplot as plt

def IQplot(samples):
    """
    Utility function to print I/Q samples
    :param samples: Complex I/Q samples to print
    """
    plt.figure(figsize=(5, 5))
    plt.scatter(samples.real, samples.imag)
    plt.show()