# 作者 : lgs
import numpy as np
import matplotlib.pyplot as plt


def PlanckL(lambda1, T):
    """出射度 W/m^2/μm"""
    c1 = 3.7417749E8  # Planck's first constant (W·m^2·μm^4)
    c2 = 1.438769E4  # Planck's second constant  (μm·K)
    l1 = c1 / np.power(lambda1, 5)  # 第一部分
    l2 = 1 / (np.exp(c2 / lambda1 / T) - 1)  # 第二部分
    M = l1 * l2
    return M


def Intensity_0(M, theta0, X0, Y0):
    """
    用于计算太阳入射到空间网格上表面的总辐射功率I0
    where：
    L:     某波长下的辐亮度
    theta0:空间网格上表面自然光入射天顶角
    X0, Y0:空间网格上表面尺寸
    """

    D_sun_earth = 1.5E11    # 日地距离
    R_sun = 6.9627E8        # 太阳半径为70Wkm
    S_upper = X0 * Y0                        # 空间网格上表面积

    E = M * (R_sun ** 2) / (D_sun_earth ** 2)

    I0 = E * S_upper

    return I0


if __name__ == '__main__':

    L = np.zeros(1000)
    lambda1 = np.linspace(8, 15, 1000)  # 波长 μm
    T = np.linspace(100, 500, 5)         # 太阳黑体温度 K
    plt.figure()
    for i in range(5):
        for j in range(1000):
            L[j] = PlanckL(lambda1[j], T[i])
        plt.plot(lambda1, L)
    plt.xlabel('lambda(um)')
    plt.ylabel('L(W/m^2/μm)')
    plt.legend(['100K', '200K', '300K', '400K', '500K', '600K', '700K', '800K', '900K', '1000K'], loc='upper right')  # 指定legend的位置右下角
    plt.title('Planck')
    plt.show()

    # theta0 = np.pi / 6  # 入射天顶角
    # X0 = 120    # 上表面尺寸
    # Y0 = 120
    # I0 = Intensity_0(L, theta0, X0, Y0)

    # print(f"{lambda1}μm波长下的单色辐射功率为: {I0} W/μm")