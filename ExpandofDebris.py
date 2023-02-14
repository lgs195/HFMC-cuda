# 作者 : lgs
import numpy as np
import math
import miepython


def ExpandofDebris(velocity0, c1, c2, c3, c4, t, target_thickness, R_interceptor, R_particle):
    """

    Args:
        velocity0: 撞击速度 m/s
        c1: 内圈x轴向速度系数
        c2: 外圈x轴向速度系数
        c3: 破碎体积扩张系数
        c4: 碎片占总破碎体积系数
        t: 撞击后某时刻 s
        target_thickness: 目标厚度 m
        R_interceptor: 拦截器半径 m
        R_particle: 碎片云粒子半径 μm

    Returns:
        R_debris_outer: 碎片云外圈半径 m
        R_debris_inner: 碎片云内圈半径 m
        rho_debris: 碎片云数密度 μm^-3

    """

    velocity1 = c1 * velocity0  # 内圈x轴向速度
    velocity2 = c2 * velocity0  # 外圈x轴向速度

    debris_x_t = velocity1 * t  # 碎片云外圈边界
    debris_thickness_x = (velocity2 - velocity1) * t  # 碎片云轴向厚度

    delta_R = 0.5 * debris_thickness_x  # 碎片云外圈与内圈半径差
    R_debris_outer = 0.5 * debris_x_t  # 碎片云外圈半径
    R_debris_inner = R_debris_outer - delta_R  # 碎片云内圈半径

    if R_debris_inner > 0.:
        R_broken = c3 * R_interceptor  # 破碎半径

        volume_target = (R_broken / 2) ** 2 * math.pi * target_thickness  # 目标破碎体积
        volume_target = c4 * volume_target  # 碎片总体积
        volume_debris = 4. / 3. * math.pi * ((R_debris_outer ** 3) - (R_debris_inner ** 3))  # 碎片云体积
        volume_particle = 4. / 3. * math.pi * (R_particle ** 3)  # 单粒子体积(μm^3)

        N_particles = volume_target / volume_particle  # 粒子数目(×10^18)

        rho_debris = N_particles / volume_debris  # 碎片云数密度 (μm^-3)
    else:
        R_debris_inner = 0.

        R_broken = c3 * R_interceptor  # 破碎半径
        volume_target = (R_broken / 2) ** 2 * math.pi * target_thickness  # 目标破碎体积
        volume_target = c4 * volume_target  # 碎片总体积
        volume_debris = 4. / 3. * math.pi * (delta_R ** 3)  # 碎片云体积
        volume_particle = 4. / 3. * math.pi * (R_particle ** 3)  # 单粒子体积(μm^3)

        N_particles = volume_target / volume_particle  # 粒子数目(×10^18)

        rho_debris = N_particles / volume_debris  # 碎片云数密度 (μm^-3)

    print("外圈半径=%5.5f;\n 内圈半径=%5.5f;\n 光学厚度=%5.11f;\n 破碎体积=%5.11f;\n" % (
        R_debris_outer, R_debris_inner, rho_debris, volume_target))

    return R_debris_outer, R_debris_inner, rho_debris, volume_target


velocity0 = 7000  # 正向撞击速度为7km/s
c1 = 0.5  # 内圈x轴向速度系数
c2 = 0.6  # 外圈x轴向速度系数
c3 = 1.5  # 破碎体积扩张系数
c4 = 0.1  # 碎片占总破碎体积系数
# velocity1 = c1 * velocity0  # 内圈x轴向速度
# velocity2 = c2 * velocity0  # 外圈x轴向速度

t = 0.002  # 时刻为撞击后1ms
# debris_x_t = velocity1 * t  # 碎片云外圈边界
# debris_thickness_x = (velocity2 - velocity1) * t  # 碎片云轴向厚度

target_thickness = 1.0  # 目标厚度为1m

# print(debris_thickness_x, debris_x_t)

# delta_R = 0.5 * debris_thickness_x         # 碎片云外圈与内圈半径差
# R_debris_outer = 0.5 * debris_x_t          # 碎片云外圈半径
# R_debris_inner = R_debris_outer - delta_R  # 碎片云内圈半径

# print("外圈半径=%5.5f;\n 内圈半径=%5.5f;\n" % (R_debris_outer, R_debris_inner))

R_interceptor = 0.25  # 拦截弹半径为25cm
# R_broken = 1.5um * R_interceptor      # 破碎半径
R_particle = 1.  # 粒子半径为1μm

# volume_target = (R_broken / 2) ** 2 * math.pi * target_thickness                     # 目标破碎体积
# volume_debris = 4. / 3. * math.pi * ((R_debris_outer ** 3) - (R_debris_inner ** 3))  # 碎片云体积
# volume_particle = 4. / 3. * math.pi * (R_particle ** 3)                              # 单粒子体积(μm^3)

# N_particles = volume_target / volume_particle  # 粒子数目(×10^18)
#
# rho_debris = N_particles / volume_debris       # 碎片云数密度 (μm^-3)

# print(rho_debris)

R_debris_outer, R_debris_inner, rho_debris, volume_target = ExpandofDebris(velocity0, c1, c2, c3, c4, t, target_thickness,
                                                                           R_interceptor, R_particle)

print("n*d=%5.8f" % (rho_debris * (R_debris_outer - R_debris_inner)))
################################################################################
"""mie散射测试"""
# m = 6.5580 + 1j * 40.185
# rho = rho_debris
# radius = R_particle
# lambda1 = 4.
# nre_med = 1.
#
# nangles = 1000
#
# x = 2 * math.pi * radius / (lambda1 / nre_med)  # 粒径参数
# A = np.pi * radius * radius
#
# mu = np.zeros(nangles)  # 角度余弦
#
# # 入射角度离散化
# for i in range(nangles):
#     mu[i] = math.cos(math.pi * i / nangles)
#
# qext, qsca, qback, g = miepython.mie(m, x)             # 消光截面、散射截面
# qabs = qext - qsca                                     # 吸收截面
# s1, s2 = miepython.mie_S1_S2(m, x, mu, norm='albedo')
#
# mus = qsca * A * rho_debris * 1e6  # 散射系数
# mua = qabs * A * rho_debris * 1e6  # 吸收系数
#
# albedo = mus / (mus - mua)  # 反照率
#
# print(mus, mua, albedo)
#
#
# print("Polarized Monte Carlo\n dia=%5.5f;\n qext=%5.5f;\n qsca=%5.5f;\n g=%5.5f;\n  "
#       "rho=%5.10f;\n mus=%5.5f;\n mua=%5.5f;\n albedo=%5.5f;\n" % (radius * 2, qext, qsca, g, rho, mus, mua, albedo))
#
# # Mueller矩阵系数
# s11 = np.zeros(nangles).astype(np.float32)
# s12 = np.zeros(nangles).astype(np.float32)
# s33 = np.zeros(nangles).astype(np.float32)
# s43 = np.zeros(nangles).astype(np.float32)
#
# # Mie Scattering Mueller Matrix
# for i in range(nangles):
#     s11[i] = 0.5 * abs(s2[i]) * abs(s2[i]) + 0.5 * abs(s1[i]) * abs(s1[i])
#     s12[i] = 0.5 * abs(s2[i]) * abs(s2[i]) - 0.5 * abs(s1[i]) * abs(s1[i])
#     s33[i] = (np.conj(s1[i]) * s2[i]).real
#     s43[i] = (np.conj(s1[i]) * s2[i]).imag
#     print("%5.5f\t %5.5f\t %5.5f\t %5.5f\t %5.5f\t %5.5f\t %5.5f\t %5.5f\n" % (
#         s11[i], s12[i], s33[i], s43[i], s1[i].real, s1[i].imag, s2[i].real, s2[i].imag))
