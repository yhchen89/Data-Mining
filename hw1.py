import sys
import pandas as pd
import numpy as np
from linear_regression import LinearRegression

# 讀入train.csv，繁體字以big5編碼
data = pd.read_csv('./train_cleaned.csv', encoding = 'big5')
# 丟棄前兩列，需要的是從第三列開始的數值
data = data.iloc[:, 3:]
# 把dataframe轉換成numpy的數組
raw_data = data.to_numpy()
month_data = {}
# 把數據整理成18 * 480的數組，一個月20天，每天24小時，一小時18個數據
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample
x = np.empty([12 * 471, 18 * 9], dtype = float) # 12 * 471個數據，每個數據有18 * 9個數據
y = np.empty([12 * 471, 1], dtype = float) # 12 * 471個數據，每個數據有1個數據
# 每個月的數據，每9小時的數據，預測第10小時的PM2.5
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] #value

model = LinearRegression()
learning_rate = 100
iter_time = 1800
model.fit(x, y, learning_rate, iter_time, normalize = True)

# 讀入測試數據test.csv
testdata = pd.read_csv('./test_cleaned.csv', header = None, encoding = 'big5')
# 丟棄前兩列，需要的是從第3列開始的數據
test_data = testdata.iloc[:, 2:]
# 將dataframe變成numpy數組
test_data = test_data.to_numpy()
# 將test數據也變成 240 個維度爲 18 * 9 + 1 的數據。
test_x = np.empty([244, 18*9], dtype = float)
for i in range(244):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)

test_x = np.concatenate((np.ones([244, 1]), test_x), axis = 1).astype(float)
#print(model.predict(test_x))
ans_y = model.predict(test_x)

import csv
with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['index', 'answer']
    print(header)
    csv_writer.writerow(header)
    for i in range(244):
        row = ['index_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)