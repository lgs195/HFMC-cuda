# coding=UTF-8
import numpy as np
import csv

theta = np.linspace(np.pi/2, np.pi, 40)

with open(f"theta.csv", 'w', newline='') as old_csv:
    # csv.writer(old_csv).writerow(key)  # 写入一行标题
    csv.writer(old_csv).writerow(theta)  # 写入多行数据
old_csv.close()