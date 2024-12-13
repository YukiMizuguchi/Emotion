import sys
sys.path.append("c:/users/_s2220459/appdata/local/packages/pythonsoftwarefoundation.python.3.12_qbz5n2kfra8p0/localcache/local-packages/python312/site-packages")

import itertools
from matplotlib import pyplot as plt
import pandas as pd

#テキスト読み込み
f = open("テキストファイルのパスを書く","r",encoding="utf-8")
#それぞれの行をまとめて取得する
data = f.readlines()#各行がリストへ格納
data = [v.replace("\n", "") for v in data]
for i,v in enumerate(data):
    data[i] = v.replace("\n", "")
    data[i]= list(map(float,(data[i].split())))


generation = int(len(data)/216)

#################################
time = 300 # 割合などを出力する世代 
#################################

# 指定した世代のタイプ分布や協力率などを出力
print("the", time, "th generation of", generation)
for i in range(len(data)):
    if data[i][0] == (time-1) and data[i][8] > 0.01:
        r1 = data[i][2]
        r2 = data[i][3]
        r3 = data[i][4]
        e1 = data[i][5]
        e2 = data[i][6]
        e3 = data[i][7]
            
        print("r(C,C):", r1, " r(D,C):", r2, " r(C,D):", r3, " e(C,C):", e1, " e(D,C):", e2, " e(C,D):", e3, " frequency:", data[i][8], " average payoff:", data[i][9], " cooperation rate:", data[i][10])
        

x = [i for i in range(350)]
t = [i for i in range(len(data))]
r1_y = [0]*2
r2_y = [0]*2
r3_y = [0]*2
e1_y = [0]*3
e2_y = [0]*3
e3_y = [0]*3

df = pd.DataFrame(data, index=t, columns=['generation', 'strategy', 'r(C,C)', 'r(D,C)', 'r(C,D)', 'e(C,C)', 'e(D,C)', 'e(C,D)', 'number', 'payoff', 'CRate', 'TotalCRate'])

r1_df = df.groupby(['generation', 'r(C,C)'], as_index=False).sum()
for i in range(2):
    r1_y[i] = r1_df[r1_df['r(C,C)'] == i]['number'].to_list()

r2_df = df.groupby(['generation', 'r(D,C)'], as_index=False).sum()
for i in range(2):
    r2_y[i] = r2_df[r2_df['r(D,C)'] == i]['number'].to_list()

r3_df = df.groupby(['generation', 'r(C,D)'], as_index=False).sum()
for i in range(2):
    r3_y[i] = r3_df[r3_df['r(C,D)'] == i]['number'].to_list()


e1_df = df.groupby(['generation', 'e(C,C)'], as_index=False).sum()
for i in range(3):
    e1_y[i] = e1_df[e1_df['e(C,C)'] == i-1]['number'].to_list()

e2_df = df.groupby(['generation', 'e(D,C)'], as_index=False).sum()
for i in range(3):
    e2_y[i] = e2_df[e2_df['e(D,C)'] == i-1]['number'].to_list()

e3_df = df.groupby(['generation', 'e(C,D)'], as_index=False).sum()
for i in range(3):
    e3_y[i] = e3_df[e3_df['e(C,D)'] == i-1]['number'].to_list()


CoopRate = df[(df['strategy'] == 215)]['TotalCRate'].to_list()

# 各rの割合と協力率の遷移グラフを出力
fig = plt.figure()
ax = fig.add_subplot(4, 1, 1)
for i in range(2):
    ax.plot(x, r1_y[i], label="r=" + str(i))
ax.set_title("r(C,C)")
ax.set_ylim(-0.1, 1.1)
ax.legend(bbox_to_anchor=(1, 1))

ax = fig.add_subplot(4, 1, 2)
for i in range(2):
    ax.plot(x, r2_y[i])
ax.set_ylim(-0.1, 1.1)
ax.set_title("r(D,C)")

ax = fig.add_subplot(4, 1, 3)
for i in range(2):
    ax.plot(x, r3_y[i])
ax.set_ylim(-0.1, 1.1)
ax.set_title("r(C,D)")

ax = fig.add_subplot(4, 1, 4)
ax.plot(x, CoopRate)
ax.set_ylim(-0.1, 1.1)
ax.set_title("C Rate")
plt.show()

# 各eの割合と協力率の遷移グラフを出力
fig = plt.figure()
ax = fig.add_subplot(4, 1, 1)
for i in range(3):
    ax.plot(x, e1_y[i], label="e=" + str(i-1))
ax.set_ylim(-0.1, 1.1)
ax.set_title("e(C,C)")
ax.legend(bbox_to_anchor=(1, 1))

ax = fig.add_subplot(4, 1, 2)
for i in range(3):
    ax.plot(x, e2_y[i])
ax.set_ylim(-0.1, 1.1)
ax.set_title("e(D,C)")

ax = fig.add_subplot(4, 1, 3)
for i in range(3):
    ax.plot(x, e3_y[i])
ax.set_ylim(-0.1, 1.1)
ax.set_title("e(C,D)")

ax = fig.add_subplot(4, 1, 4)
ax.plot(x, CoopRate)
ax.set_ylim(-0.1, 1.1)
ax.set_title("C Rate")

plt.show()

