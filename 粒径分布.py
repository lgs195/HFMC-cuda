# coding=UTF-8
import numpy as np
import matplotlib.pyplot as plt
import csv


# 对数正态分布的概率密度函数
def normpdf(x, mu, sigma):
    pdf = np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma ** 2)) / (x * sigma * np.sqrt(2 * np.pi))
    return pdf


mu1, sigma1 = 1.0, 1.4  # mu:期望;sigma:标准差
x1 = np.arange(0.1, 10.00, 0.01)  # 生成数据，步长越小，曲线越平滑
y1 = normpdf(x1, mu1, sigma1)

# mu2, sigma2 = 10.0, 1.4  # mu:期望;sigma:标准差
# x2 = np.arange(0.1, 80, 0.1)  # 生成数据，步长越小，曲线越平滑
# y2 = normpdf(x2, mu2, sigma2)
#
# mu3, sigma3 = 200.0, 1.4  # mu:期望;sigma:标准差
# x3 = np.arange(1, 800, 1)  # 生成数据，步长越小，曲线越平滑
# y3 = normpdf(x3, mu3, sigma3)

result = sum(y1*0.01)
print(result)

# key = np.array(['r', 'P'])

plt.plot(x1, y1,color='r')
# plt.plot(x2, y2,color='g')
# plt.plot(x3, y3,color='b')
# plt.xscale("log")
# plt.yscale("log")
plt.show()

# with open(f"粒径分布.csv", 'w', newline='') as old_csv:
#     csv.writer(old_csv).writerow(key)  # 写入一行标题
#     csv.writer(old_csv).writerow(x)  # 写入多行数据
#     csv.writer(old_csv).writerow(y)  # 写入多行数据
#     old_csv.close()
