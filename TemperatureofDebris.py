import numpy as np


def TofDebris(t):
    """

    Args:
        t: 冷却时间(ms)

    Returns:
        碎片云整体温度(T)

    """
    a = 4.825
    b = 1.709
    c = 0.712
    d = 1.506
    return a / (b * (t ** c) + d) * 10004.8


def TofDebris_meta(T0, epsilon, R0, sigma, Cv, mg, t):
    """

    Args:
        T0: 初始温度
        epsilon: 发射率？
        R0: 微元半径
        sigma: 发射系数 斯特藩玻尔兹曼常数 （sigmaT4）
        Cv: 比热容
        mg: 微元质量
        t: 冷却时间

    Returns:
        T: t时刻温度
    """

    temp = 1 / T0 + 3 * (epsilon * 4 * np.pi * (R0 ** 2) * sigma / Cv / mg) * t
    T = 1 / np.power(temp, 1 / 3)

    return T
