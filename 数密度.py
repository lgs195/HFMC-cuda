import numpy as np


def Dnum(X0, Y0, Z0, r, m, rho):

    V0 = X0 * Y0 * Z0  # 区域体积
    V = 0.8888 / 4 * V0  # 碎片体积

    v = 0.75 * np.pi * r * r * r # 单粒体积
    N = m / rho / v  # 碎片数

    n = N / V  # 光学厚度

    return n


def main():

    X0 = 30
    Y0 = 70
    Z0 = 30
    r = 1E-6
    m = 10
    rho = 2.7 * 1E3

    n = Dnum(X0, Y0, Z0, r, m, rho)
    print(n)


if __name__ == '__main__':
    main()