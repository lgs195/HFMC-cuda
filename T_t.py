#coding=gbk
import numpy as np
import matplotlib.pyplot as plt


def T_t(T0, epsilon, R0, sigma, CV, mg, t):
    """

    Args:
        T0: 初始温度
        epsilon: 发射率
        R0: 微元半径
        sigma: 斯特芬玻尔兹曼常数
        CV: 比热容
        mg: 微元质量
        t: 时间

    Returns: T: t时刻温度

    """
    S = 4 * np.pi * (R0 ** 2)
    temp1 = epsilon * S * sigma / (CV * mg)
    temp2 = 1 / (T0 ** 3) + 3 * temp1 * t
    T = 1 / np.power(temp2, 1 / 3)

    return T


def main():
    # 初始条件
    T0 = 3200
    epsilon = 0.06
    R0 = 1E-6
    sigma = 5.67E-8
    CV = 0.88E3
    mg = 2.7E-3 * 4 / 3 * np.pi * (R0 ** 3) * 1E6
    t = np.linspace(1, 100, 100)

    t = t * 1E-6
    print(t)

    T = np.zeros(100)
    # 温度计算
    for i in range(100):
        T[i] = T_t(T0, epsilon, R0, sigma, CV, mg, t[i])
    print(T)

    plt.figure()
    plt.plot(t, T)
    plt.show()




if __name__ == '__main__':
    main()
