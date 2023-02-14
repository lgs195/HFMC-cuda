# coding:utf-8
# 作者 : lgs
import numpy as np
import math
import miepython


def ScatParam(m, rho, radius, lambda1, nre_med, nangles):
    x = 2 * math.pi * radius / (lambda1 / nre_med)  # 粒径参数
    A = np.pi * radius * radius

    mu = np.zeros(nangles)  # 角度余弦

    # 入射角度离散化
    for i in range(nangles):
        mu[i] = math.cos(math.pi * i / nangles)

    qext, qsca, qback, g = miepython.mie(m, x)  # 消光截面、散射截面
    qabs = qext - qsca  # 吸收截面
    # s1, s2 = miepython.mie_S1_S2(m, x, mu, norm='albedo')

    mus = qsca * A * rho * 1e6  # 散射系数
    mua = qabs * A * rho * 1e6  # 吸收系数

    albedo = mus / (mus - mua)  # 反照率

    print("Polarized Monte Carlo\n dia=%5.5fμm;\n x=%5.5f;\n qext=%5.5f;\n qsca=%5.5f;\n g=%5.5f;\n "
          "rho=%5.10fμm^-3;\n mus=%5.5f;\n mua=%5.5f;\n albedo=%5.5f;\n" % (
              radius * 2, x, qext, qsca, g, rho, mus, mua, albedo))

    # Mueller矩阵系数
    s11 = np.zeros(nangles).astype(np.float32)
    s12 = np.zeros(nangles).astype(np.float32)
    s33 = np.zeros(nangles).astype(np.float32)
    s43 = np.zeros(nangles).astype(np.float32)

    # # Mie Scattering Mueller Matrix
    # for i in range(nangles):
    #     s11[i] = 0.75
    #     s12[i] = 0.5 * abs(s2[i]) * abs(s2[i]) - 0.5 * abs(s1[i]) * abs(s1[i])
    #     s33[i] = (np.conj(s1[i]) * s2[i]).real
    #     s43[i] = (np.conj(s1[i]) * s2[i]).imag
    #     print("%5.5f\t %5.5f\t %5.5f\t %5.5f\t %5.5f\t %5.5f\t %5.5f\t %5.5f\n" % (
    #         s11[i], s12[i], s33[i], s43[i], s1[i].real, s1[i].imag, s2[i].real, s2[i].imag))

    # Rayleigh Scattering Mueller Matrix
    for i in range(nangles):
        s11[i] = 0.75 * (mu[i] * mu[i] + 1.)
        s12[i] = 0.75 * (mu[i] * mu[i] - 1.)
        s33[i] = 1.5 * mu[i]
        s43[i] = 0.
        print("%5.5f\t %5.5f\t %5.5f\t %5.5f\n" % (s11[i], s12[i], s33[i], s43[i]))

    return s11, s12, s33, s43, rho, mus, mua, albedo


def ScatParam_easy(nangles):
    mu = np.zeros(nangles)  # 角度余弦

    # 入射角度离散化
    for i in range(nangles):
        mu[i] = math.cos(math.pi * i / nangles)

    # Mueller矩阵系数
    s11 = np.zeros(nangles).astype(np.float32)
    s12 = np.zeros(nangles).astype(np.float32)
    s33 = np.zeros(nangles).astype(np.float32)
    s43 = np.zeros(nangles).astype(np.float32)

    # Rayleigh Scattering Mueller Matrix
    for i in range(nangles):
        s11[i] = 0.75 * (mu[i] * mu[i] + 1.)
        s12[i] = 0.75 * (mu[i] * mu[i] - 1.)
        s33[i] = 1.5 * mu[i]
        s43[i] = 0.
        print("%5.5f\t %5.5f\t %5.5f\t %5.5f\n" % (s11[i], s12[i], s33[i], s43[i]))

    return s11, s12, s33, s43

# m = 6.5580 + 1j * 40.185
# lambda1 = 4.
# nre_med = 1.
# radius = 1.
#
# nangles = 1000
#
# rho = 1.5E-7
#
# s11, s12, s33, s43, rho, mus, mua, albedo = ScatParam(m, rho, radius, lambda1, nre_med, nangles)
