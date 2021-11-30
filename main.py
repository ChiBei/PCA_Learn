import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据
x = np.array([9, 16, 25, 14, 10, 18, 0, 16, 5, 19, 16, 20])
y = np.array([39, 57, 93, 61, 50, 75, 32, 85, 42, 70, 66, 80])
z = np.array([7, 58, 73, 1, 0, 55, 72, 87, 4, 10, 86, 30])
D0 = np.zeros([12, 3])  # 创建一个12行x3列的零矩阵
D0[:, 0] = x
D0[:, 1] = y
D0[:, 2] = z
print("原数据：\n", D0)
print("原数据的维数: \n", D0.shape)

# 中心化
D0_Av=np.mean(D0, axis=0)
print("均值：",D0_Av)
D = D0 - D0_Av
print("中心化之后：\n", D)
print("中心化数据的维数: \n", D.shape)

# 绘制中心化对比图
ax = plt.subplot(111, projection='3d')
for i in range(0, 12):
    ax.scatter(D[i][0], D[i][1], D[i][2], c='b')
for i in range(0, 12):
    ax.scatter(D0[i][0], D0[i][1], D0[i][2], c='r')
plt.title("原数据和中心化之后的数据对比(三维)")
ax.set_zlabel('Z')
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()

# 绘制中心化对比图(投在二维上)
plt.scatter(D0[:, 0], D0[:, 1], label='原数据(XY方向投影)')
plt.scatter(D[:, 0], D[:, 1], label='中心化之后(XY方向投影)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title("原数据和中心化之后的数据对比(二维)")
plt.legend()
plt.grid(True, linestyle='-.')
ax = plt.gca()  # 获取当前坐标的位置
ax.spines['right'].set_color('None')  # 去掉坐标图的上和右的黑边
ax.spines['top'].set_color('None')
ax.xaxis.set_ticks_position('bottom')  # 设置bottom为x轴
ax.yaxis.set_ticks_position('left')  # 设置left为y轴
ax.spines['bottom'].set_position(('data', 0))  # 这个位置的括号要注意
ax.spines['left'].set_position(('data', 0))
plt.show()

# 计算协方差矩阵
Cov1 = np.mat(D.T) * np.mat(D) / (len(x) - 1)
print("协方差矩阵：\n", Cov1)
print("协方差矩阵的维数: \n", Cov1.shape)

# 计算特征值、特征向量
Cov2 = np.cov(D, rowvar=False)
eigVals, eigVects = np.linalg.eig(np.mat(Cov2))
# # 排序
# eigVects = eigVects[:, np.argsort(-eigVects)]
# eigVals = eigVals[np.argsort(-eigVals)]
print("特征值: \n", eigVals)
print("特征向量的矩阵: \n", eigVects)
print("特征向量的矩阵的维数: \n", eigVects.shape)

# 统计哪些是主成分
tot = sum(eigVals)
eigVals_exp = [(i / tot) for i in sorted(eigVals, reverse=False)]  # 特征值的贡献率
cum_eigVals_exp = np.cumsum(eigVals_exp)  # 累积特征值的贡献率

# 绘制特征值的贡献率与累计贡献率，用来判断主成分
plt.bar(range(0, 3), eigVals_exp, alpha=0.25, label='贡献率')
plt.plot(range(0, 3), cum_eigVals_exp, marker='o', label='累计贡献率')
plt.ylabel('贡献率')
plt.xlabel('index')
plt.title("特征值的贡献率与累计贡献率")
plt.legend()
plt.grid(True, linestyle='-.')
plt.show()

# 截取出主成分
eigVects_main = eigVects[:, 1:3]  # 截取所有行，前1~3-1共2列，截取的后两维数据作为主成分
print("截取的特征向量的矩阵: \n", eigVects_main)
print("截取的特征向量的矩阵的维数: \n", eigVects_main.shape)
#使用截取的矩阵来降维中心化的数据
D_main = np.matmul(D, eigVects_main)
print("降维的中心化的矩阵: \n", D_main)
print("降维的中心化的矩阵的维数: \n", D_main.shape)
#中心化，恢复最原始的数据
D0_main = D_main + D0_Av[1:3]
print("降维的原矩阵: \n", D0_main)
print("降维的原矩阵的维数: \n", D0_main.shape)

# 绘制降维对比图
ax = plt.subplot(111, projection='3d')
for i in range(0, 12):
    ax.scatter(D0[i][0], D0[i][1], D0[i][2], c='b')
for i in range(0, 12):
    ax.scatter(0, D0_main[i][0], D0[i][1], c='r')
plt.title("降维对比(三维)")
ax.set_zlabel('Z')
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()
