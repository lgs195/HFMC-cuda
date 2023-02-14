# 创建一个三维粒子密度分布模型

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# import matplotlib
import matplotlib.pyplot as plt


# matplotlib.use('TkAgg')


# from matplotlib.animation import FuncAnimation
# from numba import njit, int32, float64, complex128
# from numba.cuda import jit

def BoundaryEdges(x):
    tmp1 = 2 * x * x + 1
    tmp2 = np.sqrt(8 * x * x + 1)
    y = 2 * np.sqrt(-tmp1 + tmp2)

    return y


def innerEdges(x, k):
    y = k * BoundaryEdges(x / k)

    return y


def InCloud(yy, Zmax, k):
    if -0.4 <= yy <= 1.0:
        r_out = BoundaryEdges(yy) * Zmax
        if -0.4 * k <= yy <= k:
            r_in = innerEdges(yy, k) * Zmax
        else:
            r_in = 0.
    else:
        r_out = 0.0
        r_in = 0.0
    return r_out, r_in


def DensityDistribution3D(rho, Xmin, Ymin, Zmin, Xmax, Ymax, Zmax, kk):
    X0 = Xmax - Xmin
    Y0 = Ymax - Ymin
    Z0 = Zmax - Zmin
    d = np.zeros((Y0, X0, Z0))  # (y, x, z)
    for i in range(Y0):
        for j in range(X0):
            for k in range(Z0):
                dist = np.sqrt(np.square(j + Xmin) + np.square(k + Zmin))
                yy = (i + Ymin) / Ymax
                r_out, r_in = InCloud(yy, Zmax, kk)
                if r_in < dist < r_out:
                    d[i, j, k] = rho
                    # print(rho)
                else:
                    d[i, j, k] = None
    return d


def main():
    rho = 4
    Xmin = -30
    Ymin = -40
    Zmin = -30
    Xmax = 30
    Ymax = 100
    Zmax = 30
    kk = 0.8

    X0 = Xmax - Xmin
    Y0 = Ymax - Ymin
    Z0 = Zmax - Zmin

    x = np.arange(0, X0, 1)
    y = np.arange(0, Y0, 1)
    z = np.arange(0, Z0, 1)

    x = np.arange(Xmin, Xmax, 1)
    y = np.arange(Ymin, Ymax, 1)
    z = np.arange(Zmin, Zmax, 1)

    x, y, z = np.meshgrid(x, y, z)

    D = DensityDistribution3D(rho, Xmin, Ymin, Zmin, Xmax, Ymax, Zmax, kk)
    # D = np.int32(D)
    # print(D)

    # color = np.empty((200, 100, 100))
    # for i in y:
    #     for j in x:
    #         for k in z:
    #             temp = 20000*i + 100*j + k
    #             color[i, j, k] = D[temp]

    color = np.array([50*D[i-Ymin, j-Xmin, k-Zmin] for i, j, k in zip(y, x, z)])
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(x, y, z, c=color)
    # ax.set_box_aspect((3, 7, 3))
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.view_init(340, -25)
    # plt.show()
    #
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, c=color)
    ax.set_box_aspect((3, 7, 3))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(25, 25)
    plt.show()


if __name__ == '__main__':
    main()
