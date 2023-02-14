# 作者 : lgs
##使用curve_fit

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# 自定义函数 e指数形式
def func(x, a, b, c, d):
    return a / (b * (x ** c) + d)


# 定义x、y散点坐标
x = [0., 0.02, 0.04, 0.06, 0.08, 0.1]
x = np.array(x)
num = [3.200, 3.020, 2.870, 2.750, 2.690, 2.650]
y = np.array(num)

# 非线性最小二乘法拟合
popt, pcov = curve_fit(func, x, y)
# 获取popt里面是拟合系数
print(popt)
a = popt[0]
b = popt[1]
c = popt[2]
d = popt[3]
yvals = func(x, a, b, c, d)  # 拟合y值
print('popt:', popt)
print('系数a:', a)
print('系数b:', b)
print('系数c:', c)
print('系数d:', d)
print('系数pcov:', pcov)
print('系数yvals:', yvals)
# 绘图
x0 = np.linspace(0, 30, 300)
yvals0 = func(x0, a, b, c, d)  # 拟合y值
# plot1 = plt.plot(x, y, 's', label='original values')
# plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
plot3 = plt.plot(x0, 1000 * yvals0, 'r', label='polyfit values')
plt.xlabel('Time(ms)')
plt.ylabel('Temperature(K)')
plt.legend(loc=1)  # 指定legend的位置右下角
plt.title('curve_fit')
plt.show()