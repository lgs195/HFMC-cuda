# coding=UTF-8

import numpy as np
import csv
import matplotlib.pyplot as plt
import ScatParam
import miepython


def main():
    lambda1 = 0.9       # 波长
    nangles = 1000  # 角度划分

    # 折射率实部
    # nre_p = 6.7717      # 粒子折射率实部 (Al4.0μm)
    # nre_p = 1.5237      # 粒子折射率实部 (Al1.5μm)
    # nre_p = 2.1110      # 粒子折射率实部 (Al0.9μm)
    # nre_p = 5.3121      # 粒子折射率实部 (Graphite4.0μm)
    # nre_p = 3.5824      # 粒子折射率实部 (Graphite1.5μm)
    nre_p = 3.1187      # 粒子折射率实部 (Graphite0.9μm)
    nre_med = 1.00      # 介质折射率实部

    m_real = nre_p / nre_med  # 相对折射率实部

    # 折射率虚部
    # m_imag = 38.679           # 相对折射率虚部 (Al4.0μm)
    # m_imag = 15.115           # 相对折射率虚部 (Al1.5μm)
    # m_imag = 8.2197           # 相对折射率虚部 (Al0.9μm)
    # m_imag = 4.4368           # 相对折射率虚部 (Graphite4.0μm)
    # m_imag = 2.5060           # 相对折射率虚部 (Graphite1.5μm)
    m_imag = 1.8606           # 相对折射率虚部 (Graphite0.9μm)
    m = m_real - 1j * m_imag  # 相对折射率

    radius1 = np.linspace(0.1, 10, 1000)

    albedo = np.zeros(1000)
    Qext = np.zeros(1000)
    G = np.zeros(1000)

    for i in range(1000):
        x = 2 * np.pi * radius1[i] / (lambda1 / nre_med)
        qext, qsca, qback, g = miepython.mie(m, x)
        albedo[i] = qsca / qext
        Qext[i] = qext
        G[i] = g

    key = np.array(['r', 'albedo', 'Qext'])
    with open(f"../data/Mie/Q_r/Graphite/0.9um/1.csv", 'w', newline='') as old_csv:
        csv.writer(old_csv).writerow(key)  # 写入一行标题
        csv.writer(old_csv).writerow(radius1)  # 写入一行数据
        csv.writer(old_csv).writerow(albedo)  # 写入一行数据
        csv.writer(old_csv).writerow(Qext)  # 写入一行数据
        csv.writer(old_csv).writerow(G)  # 写入一行数据
        old_csv.close()


if __name__ == '__main__':
    main()

