import numpy as np
import csv


def BoundaryEdges(x):

    tmp1 = 2 * x * x + 1
    tmp2 = np.sqrt(8 * x * x + 1)
    y = 2 * np.sqrt(-tmp1 + tmp2)

    return y


def innerEdges(x, k):

    y = k * BoundaryEdges(x / k)

    return y


def main1():

    """
    内外边界轮廓曲线
    """

    x1 = np.linspace(-0.4, 1, 1000)
    x2 = np.linspace(-0.32, 0.8, 800)

    y1 = np.zeros(1000)
    y2 = np.zeros(800)

    for i in range(1000):
        y1[i] = BoundaryEdges(x1[i])

    for j in range(800):
        y2[j] = innerEdges(x2[j], k=0.8)

    with open(f"../data/Models/boundaryedges1.csv", 'w', newline='') as old_csv:
        # csv.writer(old_csv).writerow(key)  # 写入一行标题
        csv.writer(old_csv).writerow(x1)  # 写入多行数据
        csv.writer(old_csv).writerow(x2)  # 写入多行数据
        csv.writer(old_csv).writerow(y1)  # 写入多行数据
        csv.writer(old_csv).writerow(y2)  # 写入多行数据
    old_csv.close()


def main2():

    """
    计算归一化体积
    """

    A = 0.
    x = np.linspace(0, 1, 1001)

    for i in range(1001):
        if x[i] <= 0.8:
            y1 = BoundaryEdges(x[i])
            y2 = innerEdges(x[i], k=0.8)
            A += np.pi * (y1 * y1 - y2 * y2)
        elif x[i] > 0.8:
            y1 = BoundaryEdges(x[i])
            A += np.pi * y1 * y1

    V = A * 0.001
    print(V)


if __name__ == '__main__':
    main2()