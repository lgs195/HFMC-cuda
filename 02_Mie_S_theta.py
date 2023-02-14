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
    nre_p = 2.1110      # 粒子折射率实部 (Al0.9μm)
    # nre_p = 5.3121      # 粒子折射率实部 (Graphite4.0μm)
    # nre_p = 3.5824      # 粒子折射率实部 (Graphite1.5μm)
    # nre_p = 3.1187      # 粒子折射率实部 (Graphite0.9μm)
    nre_med = 1.00      # 介质折射率实部

    m_real = nre_p / nre_med  # 相对折射率实部

    # 折射率虚部
    # m_imag = 38.679           # 相对折射率虚部 (Al4.0μm)
    # m_imag = 15.115           # 相对折射率虚部 (Al1.5μm)
    m_imag = 8.2197           # 相对折射率虚部 (Al0.9μm)
    # m_imag = 4.4368           # 相对折射率虚部 (Graphite4.0μm)
    # m_imag = 2.5060           # 相对折射率虚部 (Graphite1.5μm)
    # m_imag = 1.8606           # 相对折射率虚部 (Graphite0.9μm)
    m = m_real - 1j * m_imag  # 相对折射率

    radius1 = 1
    theta = np.zeros(nangles)  # 角度余弦
    mu = np.zeros(nangles)  # 角度余弦

    # 入射角度离散化
    for i in range(nangles):
        theta[i] = 180. * i / nangles  # theta 180
        mu[i] = np.cos(np.pi * i / nangles)  # costheta

    x = 2 * np.pi * radius1 / (lambda1 / nre_med)  # 粒径

    s1, s2 = miepython.mie_S1_S2(m, x, mu, norm='albedo')

    # Mueller矩阵系数
    s11 = np.zeros(nangles).astype(np.float32)
    s12 = np.zeros(nangles).astype(np.float32)
    s33 = np.zeros(nangles).astype(np.float32)
    s43 = np.zeros(nangles).astype(np.float32)

    # Mie Scattering Mueller Matrix
    for i in range(nangles):
        s11[i] = 0.5 * abs(s2[i]) * abs(s2[i]) + 0.5 * abs(s1[i]) * abs(s1[i])
        s12[i] = 0.5 * abs(s2[i]) * abs(s2[i]) - 0.5 * abs(s1[i]) * abs(s1[i])
        s33[i] = (np.conj(s1[i]) * s2[i]).real
        s43[i] = (np.conj(s1[i]) * s2[i]).imag

    key = np.array(['theta', 'mu', 's1', 's2', 's11', 's12', 's33', 's43'])
    with open(f"../data/Mie/S-theta/Al/0.9um/1.csv", 'w', newline='') as old_csv:
        csv.writer(old_csv).writerow(key)  # 写入一行标题
        csv.writer(old_csv).writerow(theta)  # 写入一行数据
        csv.writer(old_csv).writerow(mu)  # 写入一行数据
        csv.writer(old_csv).writerow(s1)  # 写入一行数据
        csv.writer(old_csv).writerow(s2)  # 写入一行数据
        csv.writer(old_csv).writerow(s11)  # 写入一行数据
        csv.writer(old_csv).writerow(s12)  # 写入一行数据
        csv.writer(old_csv).writerow(s33)  # 写入一行数据
        csv.writer(old_csv).writerow(s43)  # 写入一行数据
        old_csv.close()


if __name__ == '__main__':
    main()

