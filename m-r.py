# coding=UTF-8
import numpy as np


def r_m(m, rho):
    tmp = 3 * m / (4 * np.pi * rho)
    r = np.power(tmp, 1/3) * 1E4
    return r


def m_r(r, rho):
    r = r * 1E-4  # μm-cm
    m = (4/3) * np.pi * np.power(r, 3) * rho * 1E3  # g-mg
    return m


def main1():
    m = 5E-4
    rho = 2.78
    r = r_m(m, rho)
    print(r)


def main2():
    r = 1.
    rho = 2.78
    m = m_r(r, rho)
    print(m)


if __name__ == '__main__':
    main2()