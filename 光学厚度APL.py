import numpy as np
import matplotlib.pyplot as plt


def OpticalDepth(Ratio):
    Aeff = 2 / np.pi * np.arctan(1.6616 * Ratio)
    return Aeff


def main():
    Ratio = np.linspace(1, 2000, 2000)
    Ratio = Ratio / 100
    Aeff = np.zeros(2000).astype(np.float32)
    for i in range(2000):
        Aeff[i] = OpticalDepth(Ratio[i])
    plt.figure()
    plt.plot(Ratio, Aeff, linewidth=2, color='g')
    plt.xlabel('Cloud Area / Total Partical Area')
    plt.ylabel('Effective Emission Area Scaling Factor')
    plt.show()

if __name__ == '__main__':
    main()