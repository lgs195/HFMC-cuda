#coding=gbk
import numpy as np
import matplotlib.pyplot as plt


def T_t(T0, epsilon, R0, sigma, CV, mg, t):
    """

    Args:
        T0: ��ʼ�¶�
        epsilon: ������
        R0: ΢Ԫ�뾶
        sigma: ˹�طҲ�����������
        CV: ������
        mg: ΢Ԫ����
        t: ʱ��

    Returns: T: tʱ���¶�

    """
    S = 4 * np.pi * (R0 ** 2)
    temp1 = epsilon * S * sigma / (CV * mg)
    temp2 = 1 / (T0 ** 3) + 3 * temp1 * t
    T = 1 / np.power(temp2, 1 / 3)

    return T


def main():
    # ��ʼ����
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
    # �¶ȼ���
    for i in range(100):
        T[i] = T_t(T0, epsilon, R0, sigma, CV, mg, t[i])
    print(T)

    plt.figure()
    plt.plot(t, T)
    plt.show()




if __name__ == '__main__':
    main()
