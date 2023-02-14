# 作者 : lgs
from __future__ import print_function, absolute_import
import math
import time
import miepython
import numpy as np
# import matplotlib.pyplot as plt
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import csv
import ScatParam

@cuda.jit
def launch_1photon(rng_states, U0, S0, s11_1, s12_1, s33_1, s43_1, nphotons, out_x, out_y, out_phi, out_theta,
                   I_eff, Q_eff, U_eff, V_eff, n_scat, mus, mua, albedo):
    """模拟一个光子的入射的随机过程"""

    thread_id = cuda.grid(1)

    # GPU多线程
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < nphotons:

        # print(idx)

        # 初始化方向矢量
        U10 = U0[0]
        U11 = U0[1]
        U12 = U0[2]

        # 模型必要参数定义
        ALIVE = 1  # 存活
        DEAD = 0  # 死亡
        Xmax = 100   # 边界
        Xmin = -100
        Y0 = 100
        Ymax = 300
        Ymin = -100
        Zmax = 30
        Zmin = -30

        nangles = 1000   # 角度离散程度
        scat_flag = 0    # 散射判断
        absorb_flag = 0  # 吸收判断

        THRESHOLD = 0.001  # 光子存活判断
        CHANCE = 0.1

        # 上表面随机取入射点
        rndx = xoroshiro128p_uniform_float32(rng_states, thread_id)
        rndy = xoroshiro128p_uniform_float32(rng_states, thread_id)
        x = 60.0 * rndx - 30.
        y = 140.0 * rndy - 40.
        z = -30.0

        # 初始化Stokes存储器1
        S10 = S0[0]
        S11 = S0[1]
        S12 = S0[2]
        S13 = S0[3]

        # 初始化上表面Stokes矢量输出
        # costheta_u[idx] = 0.
        out_theta[idx] = 0.
        out_phi[idx] = 0.
        I_eff[idx] = 0.
        Q_eff[idx] = 0.
        U_eff[idx] = 0.
        V_eff[idx] = 0.

        # 初始化光束状态
        photon_status = ALIVE
        W = 1  # 能量权重

        # 判断是否进入介质
        inmedia = 0

        ls = 0.  # 散射间累计光学厚度

        rnd1 = 0.

        scat = 0

        """ 光束入射 """
        while photon_status == ALIVE:

            xi = math.floor(x)  # 从位置坐标获得体元索引
            yi = math.floor(y)
            zi = math.floor(z)
            xi = int(xi)
            yi = int(yi)
            zi = int(zi)

            # 获取体元索引temp
            # temp = Y0 * Z0 * xi + Z0 * yi + zi
            # temp = int(temp)

            # 这里暂时用统计方法代替 ———— 满足吸收条件直接使光子失效
            # absorb = W * (1 - albedo)       # 吸收率
            # W -= absorb                     # 能量权重

            # 粒子作用判断 如果发生作用，累计光程清零，重新得到作用距离阈值
            if ls == 0.:
                rnd1 = 0.0  # 判断穿透随机数
                while rnd1 == 0.0:
                    rnd1 = xoroshiro128p_uniform_float32(rng_states, thread_id)
            rnd2 = 0.0  # 判断吸收随机数
            while rnd2 == 0.0:
                rnd2 = xoroshiro128p_uniform_float32(rng_states, thread_id)

########################################################################################################################
                                               ### 光子透射与散射过程 ###
########################################################################################################################

            # 计算网格运动步长
            ux = U10
            uy = U11
            uz = U12

            """体元的三个法线向量"""
            A0 = 1.0
            A1 = 0.0
            A2 = 0.0

            B0 = 0.0
            B1 = 1.0
            B2 = 0.0

            C0 = 0.0
            C1 = 0.0
            C2 = 1.0

            if ux > 0:
                D0 = float(math.floor(x + 1))
            elif ux < 0:
                D0 = float(math.ceil(x - 1))
            else:
                D0 = x

            if uy > 0:
                D1 = float(math.floor(y + 1))
            elif uy < 0:
                D1 = float(math.ceil(y - 1))
            else:
                D1 = y

            if uz > 0:
                D2 = float(math.floor(z + 1))
            elif uz < 0:
                D2 = float(math.ceil(z - 1))
            else:
                D2 = z

            # 赋值 2.0 并非真值，只是为了使之大于网格内最大距离，保证距离 0.0 不会传入s
            if (ux * A0 + uy * B0 + uz * C0) == 0:
                t10 = 2.0
            else:
                t10 = (D0 - (x * A0 + y * B0 + z * C0)) / (
                        ux * A0 + uy * B0 + uz * C0)

            if (ux * A1 + uy * B1 + uz * C1) == 0:
                t11 = 2.0
            else:
                t11 = (D1 - (x * A1 + y * B1 + z * C1)) / (
                        ux * A1 + uy * B1 + uz * C1)

            if (ux * A2 + uy * B2 + uz * C2) == 0:
                t12 = 2.0
            else:
                t12 = (D2 - (x * A2 + y * B2 + z * C2)) / (
                        ux * A2 + uy * B2 + uz * C2)

            # 选取 t 的最小值存入 s
            s = t10
            if s >= t11:
                s = t11
            if s >= t12:
                s = t12

            inmedia = 0

            # 判断是否在介质内
            yy = y / Y0  # 归一化
            tmp1 = 2 * yy * yy + 1
            tmp2 = math.sqrt(8 * yy * yy + 1)
            tmp3 = 2 * math.sqrt(-tmp1 + tmp2)  # 外边界到y轴距离
            r_out = tmp3 * Zmax

            k = 0.8
            yk = yy / k
            tmp1k = 2 * yk * yk + 1
            tmp2k = math.sqrt(8 * yk * yk + 1)
            tmp3k = k * 2 * math.sqrt(-tmp1k + tmp2k)  # 内边界到y轴距离
            r_in = tmp3k * Zmax

            if r_in ** 2 <= (x * x + z * z) <= r_out ** 2:
                inmedia = 1
            else:
                inmedia = 0

            # 不在介质内
            if inmedia == 0:
                x += U10 * s
                y += U11 * s
                z += U12 * s
                x = round(x, 3)
                y = round(y, 3)
                z = round(z, 3)
            # 如果满足布格尔定律则粒子透射
            elif rnd1 < math.exp(-s * (mus + mua) - ls):
                # 记录上一次的坐标和运动距离
                x += U10 * s
                y += U11 * s
                z += U12 * s
                x = round(x, 3)
                y = round(y, 3)
                z = round(z, 3)

                ls += s * (mus + mua)  # 这里要注意：如果此cell没有介质，那么mus、mua应该为0，即没有光学厚度
            # 如果随机数2大于反照率，则被吸收
            elif rnd2 > albedo:
                absorb_flag = 1

            # 否则发生散射
            else:
                scat_flag = 1  # scat为散射标志，变为1
                scat += 1
                # s = -(1 / mus) * math.log(1 - rnd3 * (1 - math.exp(-mus * s)))  # Lsca
                s = (1 / (mus + mua)) * (math.log(1 / rnd1) - ls)  # Lsca
                ls = 0.
                x += U10 * s
                y += U11 * s
                z += U12 * s
                x = round(x, 3)
                y = round(y, 3)
                z = round(z, 3)
                while True:
                    rnd = xoroshiro128p_uniform_float32(rng_states, thread_id)
                    theta = math.acos(2 * rnd - 1)
                    # theta = rnd * np.pi  # 另一种取theta的方法

                    rnd = xoroshiro128p_uniform_float32(rng_states, thread_id)
                    phi = rnd * 2.0 * math.pi

                    ithedeg = math.floor(theta * nangles / math.pi)

                    ithedeg = int(ithedeg)

                    I0 = s11_1[0] * S10 + s12_1[0] * (S11 * math.cos(2 * phi) + S12 * math.sin(2 * phi))
                    I1 = s11_1[ithedeg] * S10 + s12_1[ithedeg] * (S11 * math.cos(2 * phi) + S12 * math.sin(2 * phi))
                    s11 = s11_1[ithedeg]
                    s12 = s12_1[ithedeg]
                    s33 = s33_1[ithedeg]
                    s43 = s43_1[ithedeg]

                    rnd = xoroshiro128p_uniform_float32(rng_states, thread_id)
                    if rnd * I0 < I1:
                        break

                # 引入入射方向矢量
                ux = U10
                uy = U11
                uz = U12

                # updataU
                costheta = math.cos(theta)
                sintheta = math.sqrt(1.0 - costheta * costheta)
                cosphi = math.cos(phi)
                if phi < math.pi:
                    sinphi = math.sqrt(1.0 - cosphi * cosphi)
                else:
                    sinphi = -math.sqrt(1.0 - cosphi * cosphi)

                if (1 - abs(uz)) <= 1.0E-12:
                    uxx = sintheta * cosphi
                    uyy = sintheta * sinphi
                    uzz = costheta * math.copysign(1, uz)
                else:
                    temp = math.sqrt(1.0 - uz * uz)
                    uxx = sintheta * (ux * uz * cosphi - uy * sinphi) / temp + ux * costheta
                    uyy = sintheta * (uy * uz * cosphi + ux * sinphi) / temp + uy * costheta
                    uzz = -sintheta * cosphi * temp + uz * costheta
                U20 = uxx
                U21 = uyy
                U22 = uzz

                # rotSphi
                cos2phi = math.cos(2 * phi)
                sin2phi = math.sin(2 * phi)

                S20 = S10
                S21 = S11 * cos2phi + S12 * sin2phi
                S22 = -S11 * sin2phi + S12 * cos2phi
                S23 = S13

                # Mueller Matrix Rotation
                S10 = s11 * S20 + s12 * S21

                S11 = s12 * S20 + s11 * S21

                S12 = s33 * S22 + s43 * S23

                S13 = -s43 * S22 + s33 * S23

                # Final Rotation
                temp = (math.sqrt(1 - costheta * costheta) * math.sqrt(1 - U22 * U22))

                if temp == 0:
                    cosi = 0
                else:
                    if (phi > math.pi) & (phi < 2 * math.pi):
                        cosi = (U22 * costheta - U12) / temp
                    else:
                        cosi = -(U22 * costheta - U12) / temp
                    if cosi > 1:
                        cosi = 1
                    if cosi < -1:
                        cosi = -1

                sini = math.sqrt(1.0 - cosi * cosi)

                cos22 = 2 * cosi * cosi - 1

                sin22 = 2 * sini * cosi

                S20 = S10

                S21 = (S11 * cos22 - S12 * sin22)

                S22 = (S11 * sin22 + S12 * cos22)

                S23 = S13

                # S1最终更新
                S11 = S21 / S20
                S12 = S22 / S20
                S13 = S23 / S20
                S10 = 1.0

                # U1最终更新
                U10 = U20
                U11 = U21
                U12 = U22

########################################################################################################################
                                                ### 光子位置判断 ###
########################################################################################################################

            if absorb_flag == 1:
            # if z >= Zmax or absorb_flag == 1:
                # 被吸收

                # 统计参数
                out_x[idx] = 0.0
                out_y[idx] = 0.0
                out_phi[idx] = 0.0
                out_theta[idx] = 0.0
                I_eff[idx] = -1.0  # 用-1.0判断是否吸收
                Q_eff[idx] = 0.0
                U_eff[idx] = 0.0
                V_eff[idx] = 0.0
                n_scat[idx] = 0.0

                photon_status = DEAD
            elif Zmax <= z or z <= Zmin or Xmax <= x or x <= Xmin or Ymax <= y or y <= Ymin:
                # 射出，统计出射光子的各参数

                # out_x[idx] = x  # 上表面出射位置坐标
                # out_y[idx] = y
                # out_phi[idx] = round(math.atan2(U10, U11), 3)  # 上表面出射角度
                # out_theta[idx] = round(math.acos(U12), 3)  # 折射定律
                # I_eff[idx] = round(S10, 3)  # 出射Stokes矢量参数
                # Q_eff[idx] = round(S11, 3)
                # U_eff[idx] = round(S12, 3)
                # V_eff[idx] = round(S13, 3)
                # n_scat[idx] = scat

                if scat_flag == 1:
                    # 统计参数
                    out_x[idx] = x                    # 上表面出射位置坐标
                    out_y[idx] = y
                    out_phi[idx] = round(math.atan2(U10, U11), 3)  # 上表面出射角度
                    out_theta[idx] = round(math.acos(U12), 3)  # 折射定律
                    I_eff[idx] = round(S10, 3)                   # 出射Stokes矢量参数
                    Q_eff[idx] = round(S11, 3)
                    U_eff[idx] = round(S12, 3)
                    V_eff[idx] = round(S13, 3)
                    n_scat[idx] = scat
                else:
                    out_x[idx] = 0.0
                    out_y[idx] = 0.0
                    out_phi[idx] = 0.0
                    out_theta[idx] = 0.0
                    I_eff[idx] = 0.0
                    Q_eff[idx] = 0.0
                    U_eff[idx] = 0.0
                    V_eff[idx] = 0.0
                    n_scat[idx] = 0.0

                photon_status = DEAD
            else:
                # 未出射，继续追踪光子

                photon_status = ALIVE


def main():
    Nabs = np.zeros(9)
    # # 光学厚度
    # for j in range(5):
    #     rj = j + 6
    #     rho = np.power(0.1, rj)
    # # 波长
    # L_lambda = [0.8, 0.9, 1.0, 1.5, 1.8, 2.0, 4.0, 5.0, 6.0]
    # L_m_real = [2.7673, 2.1110, 1.4358, 1.5237, 1.9745, 2.3493, 6.7717, 9.1528, 11.681]
    # L_m_imag = [8.3543, 8.2197, 9.4953, 15.115, 18.283, 20.309, 38.679, 47.199, 55.516]
    # for j in range(9):
    #     lambda1 = L_lambda[j]
    #     nre_p = L_m_real[j]
    #     m_imag = L_m_imag[j]
    # 半径
    L_radius = [0.8, 0.9, 1.0, 1.5, 1.8, 2.0, 3.0, 4.0, 5.0]
    for ir in range(9):
        radius1 = L_radius[ir]
        """parameters"""
        X0 = 100  # 网格划分(视野大小)
        Y0 = 100
        Z0 = 10

        # 折射率实部
        nre_p = 6.7717      # 粒子折射率实部 (Al4.0μm)
        # nre_p = 1.5237      # 粒子折射率实部 (Al1.5μm)
        # nre_p = 2.1110  # 粒子折射率实部 (Al0.9μm)
        # nre_p = 5.3121      # 粒子折射率实部 (Graphite4.0μm)
        # nre_p = 3.5824      # 粒子折射率实部 (Graphite1.5μm)
        # nre_p = 3.1187      # 粒子折射率实部 (Graphite0.9μm)
        nre_med = 1.00  # 介质折射率实部

        m_real = nre_p / nre_med  # 相对折射率实部

        # 折射率虚部
        m_imag = 38.679           # 相对折射率虚部 (Al4.0μm)
        # m_imag = 15.115           # 相对折射率虚部 (Al1.5μm)
        # m_imag = 8.2197  # 相对折射率虚部 (Al0.9μm)
        # m_imag = 4.4368           # 相对折射率虚部 (Graphite4.0μm)
        # m_imag = 2.5060           # 相对折射率虚部 (Graphite1.5μm)
        # m_imag = 1.8606           # 相对折射率虚部 (Graphite0.9μm)

        # KEY PARAMETER
        rho = 1E-8 / np.power(radius1, 3)
        lambda1 = 4.0  # 波长
        m = m_real - 1j * m_imag  # 相对折射率
        # radius1 = 2. / 2  # 粒子半径

        nphotons = 20000000  # 入射光子数
        nangles = 1000  # 角度划分

    ###############################################################
        U0 = np.zeros(3).astype(np.float32)  # 入射方向矢量
        S0 = np.zeros(4).astype(np.float32)  # 入射Stokes矢量

        s11_1, s12_1, s33_1, s43_1, qsca_1, qabs_1, mus_1, mua_1, albedo_1 = ScatParam.ScatParam(m, rho, radius1,
                                                                                                 lambda1, nre_med,
                                                                                                 nangles)

        A1 = np.pi * radius1 * radius1

        # 散射系数与吸收系数估计
        mus = qsca_1 * A1 * rho * 1e6  # 散射系数  m^-1
        mua = qabs_1 * A1 * rho * 1e6  # 吸收系数  m^-1

        # print(mus, mua)

        albedo = mus / (mus + mua)  # 反照率

        # 初始入射方向
        # U0[0] = 0.0  # 30°
        # U0[1] = 0.5
        # U0[2] = np.sqrt(3) / 2

        U0[0] = 0.0  # 0°
        U0[1] = 0.0
        U0[2] = 1.

        # U0[0] = 0.0  # 60°
        # U0[1] = np.sqrt(3) / 2
        # U0[2] = 0.5

        # 初始入射Stokes矢量
        jjj = 5
        if jjj == 1:
            S0[0] = 1
            S0[1] = 1
            S0[2] = 0
            S0[3] = 0
            print("launch H\n")

        if jjj == 2:
            S0[0] = 1
            S0[1] = -1
            S0[2] = 0
            S0[3] = 0
            print("launch V\n")

        if jjj == 3:
            S0[0] = 1
            S0[1] = 0
            S0[2] = 1
            S0[3] = 0
            print("launch P\n")

        if jjj == 4:
            S0[0] = 1
            S0[1] = 0
            S0[2] = 0
            S0[3] = 1
            print("launch R\n")

        if jjj == 5:  # 自然光
            S0[0] = 1
            S0[1] = 0
            S0[2] = 0
            S0[3] = 0
            print("launch N\n")

        # 为函数输入常量分配gpu内存
        # d0_device = cuda.to_device(d0)  # 数密度分布
        U0_device = cuda.to_device(U0)  # 初始入射方向
        S0_device = cuda.to_device(S0)  # 初始入射Stokes矢量

        s11_1_device = cuda.to_device(s11_1)  # Mueller矩阵元素s11
        s12_1_device = cuda.to_device(s12_1)  # Mueller矩阵元素s12
        s33_1_device = cuda.to_device(s33_1)  # Mueller矩阵元素s33
        s43_1_device = cuda.to_device(s43_1)  # Mueller矩阵元素s43

        x_device = cuda.device_array(nphotons)  # 出射phi存储器
        y_device = cuda.device_array(nphotons)  # 出射phi存储器
        phi_device = cuda.device_array(nphotons)  # 出射phi存储器
        theta_device = cuda.device_array(nphotons)  # 出射costheta存储器
        I_device = cuda.device_array(nphotons)  # Stokes_I存储器
        Q_device = cuda.device_array(nphotons)  # Stokes_Q存储器
        U_device = cuda.device_array(nphotons)  # Stokes_U存储器
        V_device = cuda.device_array(nphotons)  # Stokes_V存储器
        n_scat = cuda.device_array(nphotons)  # Stokes_V存储器

        threads_per_block = 512
        blocks_per_grid = math.ceil(nphotons / threads_per_block)
        rng_states = create_xoroshiro128p_states(threads_per_block * blocks_per_grid, seed=1)

        start = time.time()

        launch_1photon[blocks_per_grid, threads_per_block](rng_states, U0_device, S0_device, s11_1_device,
                                                           s12_1_device, s33_1_device, s43_1_device, nphotons, x_device,
                                                           y_device, phi_device, theta_device, I_device, Q_device,
                                                           U_device, V_device, n_scat, mus, mua, albedo)
        cuda.synchronize()
        print("gpu vector add time " + str(time.time() - start))

        out_x = x_device.copy_to_host()
        out_y = y_device.copy_to_host()
        out_phi = phi_device.copy_to_host()
        out_theta = theta_device.copy_to_host()
        out_I = I_device.copy_to_host()
        out_Q = Q_device.copy_to_host()
        out_U = U_device.copy_to_host()
        out_V = V_device.copy_to_host()
        out_scat = n_scat.copy_to_host()

        out_csv = np.zeros((nphotons, 9)).astype(np.float32)
        out_csv[:, 0] = out_x
        out_csv[:, 1] = out_y
        out_csv[:, 2] = out_phi
        out_csv[:, 3] = out_theta
        out_csv[:, 4] = out_I
        out_csv[:, 5] = out_Q
        out_csv[:, 6] = out_U
        out_csv[:, 7] = out_V
        out_csv[:, 8] = out_scat

        delete_list = []

        # 删除I=0的数据
        n_absorb = 0  # 吸收光子数
        for i in range(nphotons):
            row = out_csv[i, :]
            if row[4] == 0.:
                delete_list.append(i)
            if row[4] == -1.:
                delete_list.append(i)
                n_absorb += 1
        Nabs[ir] = n_absorb / nphotons
        print(n_absorb / nphotons)

        data = np.delete(out_csv, delete_list, axis=0)

        key1 = np.array(['x', 'y', '方位角phi', '天顶角theta', 'I', 'Q', 'U', 'V', 'n_scat'])

        # with open(f"../data/球壳MC/Al/0in/4um/r1/光学厚度/1E-{rj}.csv", 'w', newline='') as old_csv:
        #     csv.writer(old_csv).writerow(key)  # 写入一行标题
        #     csv.writer(old_csv).writerows(data)  # 写入多行数据
        #     old_csv.close()
        with open(f"../data/球壳MC/Al/0in/4um/半径/{radius1}um.csv", 'w', newline='') as old_csv:
            csv.writer(old_csv).writerow(key1)  # 写入一行标题
            csv.writer(old_csv).writerows(data)  # 写入多行数据
            old_csv.close()

    key2 = np.array(['0.8um', '0.9um', '1.0um', '1.5um', '1.8um', '2.0um', '3.0um', '4.0um', '5.0um'])
    with open(f"../data/球壳MC/Al/0in/4um/半径/absorb_rate.csv", 'w', newline='') as old_csv:
        csv.writer(old_csv).writerow(key2)  # 写入一行标题
        csv.writer(old_csv).writerow(Nabs)  # 写入多行数据
        old_csv.close()

    # for t_idx in range(1, 21):
    #
    #     # t = 3 * t_idx  # 1ms
    #
    #     # rho = 1.E-5 / (t ** 2)  # 光学厚度(μm^-3)
    #
    #     rho = 5.E-11 * t_idx  # @d=10m
    #
    #     U0 = np.zeros(3).astype(np.float32)  # 入射方向矢量
    #     S0 = np.zeros(4).astype(np.float32)  # 入射Stokes矢量
    #
    #     s11_1, s12_1, s33_1, s43_1, qsca_1, qabs_1, mus_1, mua_1, albedo_1 = ScatParam.ScatParam(m, rho, radius1,
    #                                                                                              lambda1, nre_med,
    #                                                                                              nangles)
    #
    #     A1 = np.pi * radius1 * radius1
    #
    #     # 散射系数与吸收系数估计
    #     mus = qsca_1 * A1 * rho * 1e6  # 散射系数  m^-1
    #     mua = qabs_1 * A1 * rho * 1e6  # 吸收系数  m^-1
    #
    #     albedo = mus / (mus + mua)  # 反照率
    #
    #     # 初始入射方向
    #     # U0[0] = 0.0  # 30°
    #     # U0[1] = 0.5
    #     # U0[2] = np.sqrt(3) / 2
    #
    #     U0[0] = 0.0  # 0°
    #     U0[1] = 0.0
    #     U0[2] = 1.
    #
    #     # U0[0] = 0.0  # 60°
    #     # U0[1] = np.sqrt(3) / 2
    #     # U0[2] = 0.5
    #
    #     # 初始入射Stokes矢量
    #     jjj = 5
    #     if jjj == 1:
    #         S0[0] = 1
    #         S0[1] = 1
    #         S0[2] = 0
    #         S0[3] = 0
    #         print("launch H\n")
    #
    #     if jjj == 2:
    #         S0[0] = 1
    #         S0[1] = -1
    #         S0[2] = 0
    #         S0[3] = 0
    #         print("launch V\n")
    #
    #     if jjj == 3:
    #         S0[0] = 1
    #         S0[1] = 0
    #         S0[2] = 1
    #         S0[3] = 0
    #         print("launch P\n")
    #
    #     if jjj == 4:
    #         S0[0] = 1
    #         S0[1] = 0
    #         S0[2] = 0
    #         S0[3] = 1
    #         print("launch R\n")
    #
    #     if jjj == 5:  # 自然光
    #         S0[0] = 1
    #         S0[1] = 0
    #         S0[2] = 0
    #         S0[3] = 0
    #         print("launch N\n")
    #
    #     # 为函数输入常量分配gpu内存
    #     # d0_device = cuda.to_device(d0)  # 数密度分布
    #     U0_device = cuda.to_device(U0)  # 初始入射方向
    #     S0_device = cuda.to_device(S0)  # 初始入射Stokes矢量
    #
    #     s11_1_device = cuda.to_device(s11_1)  # Mueller矩阵元素s11
    #     s12_1_device = cuda.to_device(s12_1)  # Mueller矩阵元素s12
    #     s33_1_device = cuda.to_device(s33_1)  # Mueller矩阵元素s33
    #     s43_1_device = cuda.to_device(s43_1)  # Mueller矩阵元素s43
    #
    #     x_device = cuda.device_array(nphotons)  # 出射phi存储器
    #     y_device = cuda.device_array(nphotons)  # 出射phi存储器
    #     phi_device = cuda.device_array(nphotons)  # 出射phi存储器
    #     theta_device = cuda.device_array(nphotons)  # 出射costheta存储器
    #     I_device = cuda.device_array(nphotons)  # Stokes_I存储器
    #     Q_device = cuda.device_array(nphotons)  # Stokes_Q存储器
    #     U_device = cuda.device_array(nphotons)  # Stokes_U存储器
    #     V_device = cuda.device_array(nphotons)  # Stokes_V存储器
    #     n_scat = cuda.device_array(nphotons)  # Stokes_V存储器
    #
    #     threads_per_block = 512
    #     blocks_per_grid = math.ceil(nphotons / threads_per_block)
    #     rng_states = create_xoroshiro128p_states(threads_per_block * blocks_per_grid, seed=1)
    #
    #     start = time.time()
    #
    #     launch_1photon[blocks_per_grid, threads_per_block](rng_states, U0_device, S0_device, s11_1_device,
    #                                                        s12_1_device, s33_1_device, s43_1_device, nphotons, x_device,
    #                                                        y_device, phi_device, theta_device, I_device, Q_device,
    #                                                        U_device, V_device, n_scat, mus, mua, albedo)
    #     cuda.synchronize()
    #     print("gpu vector add time " + str(time.time() - start))
    #
    #     out_x = x_device.copy_to_host()
    #     out_y = y_device.copy_to_host()
    #     out_phi = phi_device.copy_to_host()
    #     out_theta = theta_device.copy_to_host()
    #     out_I = I_device.copy_to_host()
    #     out_Q = Q_device.copy_to_host()
    #     out_U = U_device.copy_to_host()
    #     out_V = V_device.copy_to_host()
    #     out_scat = n_scat.copy_to_host()
    #
    #     out_csv = np.zeros((nphotons, 9)).astype(np.float32)
    #     out_csv[:, 0] = out_x
    #     out_csv[:, 1] = out_y
    #     out_csv[:, 2] = out_phi
    #     out_csv[:, 3] = out_theta
    #     out_csv[:, 4] = out_I
    #     out_csv[:, 5] = out_Q
    #     out_csv[:, 6] = out_U
    #     out_csv[:, 7] = out_V
    #     out_csv[:, 8] = out_scat
    #
    #     delete_list = []
    #
    #     # 删除I=0的数据
    #     for i in range(nphotons):
    #         row = out_csv[i, :]
    #         if row[4] == 0.:
    #             delete_list.append(i)
    #
    #     data = np.delete(out_csv, delete_list, axis=0)
    #
    #     key = np.array(['x', 'y', '方位角phi', '天顶角theta', 'I', 'Q', 'U', 'V', 'n_scat'])
    #
    #     with open(f"../data/球壳MC/Al/0in/0.9um/r1/9-10/result_data_{t_idx}-5E-11.csv", 'w', newline='') as old_csv:
    #         csv.writer(old_csv).writerow(key)  # 写入一行标题
    #         csv.writer(old_csv).writerows(data)  # 写入多行数据
    #         old_csv.close()


if __name__ == '__main__':
    main()
