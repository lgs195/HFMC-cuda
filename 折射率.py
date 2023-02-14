# coding=UTF-8

import numpy as np
import csv

data = np.loadtxt("E:\paper\subfile/4_Result\data\Al.txt")

key = np.array(['lambda', 'n', 'k'])

with open(f"E:\paper\subfile/4_Result\data\Al.csv", 'w', newline='') as old_csv:
    csv.writer(old_csv).writerow(key)  # 写入一行标题
    csv.writer(old_csv).writerows(data)  # 写入多行数据
    old_csv.close()