import numpy as np
import miepython
import csv

def main():
    lambda1 = 0.9  # 波长
    nangles = 1000  # 角度划分

    # 折射率实部
    # nre_p = 6.7717      # 粒子折射率实部 (Al4.0μm)
    # nre_p = 1.5237      # 粒子折射率实部 (Al1.5μm)
    nre_p = 2.1110  # 粒子折射率实部 (Al0.9μm)
    # nre_p = 5.3121      # 粒子折射率实部 (Graphite4.0μm)
    # nre_p = 3.5824      # 粒子折射率实部 (Graphite1.5μm)
    # nre_p = 3.1187      # 粒子折射率实部 (Graphite0.9μm)
    nre_med = 1.00  # 介质折射率实部

    m_real = nre_p / nre_med  # 相对折射率实部

    # 折射率虚部
    # m_imag = 38.679           # 相对折射率虚部 (Al4.0μm)
    # m_imag = 15.115           # 相对折射率虚部 (Al1.5μm)
    m_imag = 8.2197  # 相对折射率虚部 (Al0.9μm)
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

    ##### 计算偏振 #####

    S20 = 1
    S21 = 0
    S22 = 0
    S23 = 0

    # Mueller矩阵系数
    Q = np.zeros(nangles).astype(np.float32)
    U = np.zeros(nangles).astype(np.float32)
    V = np.zeros(nangles).astype(np.float32)
    DoLP = np.zeros(nangles).astype(np.float32)

    for i in range(nangles):
        # Mueller Matrix Rotation
        S10 = s11[i] * S20 + s12[i] * S21
        S11 = s12[i] * S20 + s11[i] * S21
        S12 = s33[i] * S22 + s43[i] * S23
        S13 = -s43[i] * S22 + s33[i] * S23
        Q[i] = S11 / S10
        U[i] = S12 / S10
        V[i] = S13 / S20
        DoLP[i] = np.sqrt(Q[i] ** 2 + U[i] ** 2) / 1.

    key = np.array(['theta', 'mu', 'I', 'Q', 'U', 'V', 'DoLP'])
    with open(f"../data/Mie/P-theta/Al/0.9um/1.csv", 'w', newline='') as old_csv:
        csv.writer(old_csv).writerow(key)  # 写入一行标题
        csv.writer(old_csv).writerow(theta)  # 写入一行数据
        csv.writer(old_csv).writerow(mu)  # 写入一行数据
        csv.writer(old_csv).writerow(Q)  # 写入一行数据
        csv.writer(old_csv).writerow(U)  # 写入一行数据
        csv.writer(old_csv).writerow(V)  # 写入一行数据
        csv.writer(old_csv).writerow(DoLP)  # 写入一行数据
        old_csv.close()


if __name__ == '__main__':
    main()
