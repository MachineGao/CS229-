import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##误差函数计算
def ComputeError(X,y,theta):
    inner = np.power(X@(theta.T) - y,2)
    return np.sum(inner) / 2 / X.shape[0]

##梯度下降
def GradientDiscent(X,y,theta,alpha,epoch):
    m = X.shape[0]
    cost = np.zeros(epoch)
    for i in range(epoch):
        theta = theta - alpha / m *(X@(theta.T) - y).T@X
        cost[i] = ComputeError(X,y,theta)
    return theta,cost

##正则方程
def NormalEqn(X,y):
    theta = np.linalg.inv(X.T@X)@X.T@y
    return theta

##读取数据
path = "ex1data2.txt"
data = pd.read_csv(path,header = None,names = ['Size','Bedrooms','Price'])
data.Size = (data.Size - data.Size.mean())/data.Size.std()      # 特征归一化处理
data.Bedrooms = (data.Bedrooms - data.Bedrooms.mean())/data.Bedrooms.std()
data.plot(kind = 'scatter',x = 'Size',y = 'Bedrooms',figsize = (8,5))

data.insert(0,'Ones',1) #插入1
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

X = np.array(X)
y = np.array(y)

##赋初值
theta = np.zeros(X.shape[1])
epoch = 1000
alpha = [0.01,0.03,0.1,0.3,1]

for i in range(len(alpha)):
    ##最终计算
    final_theta, cost = GradientDiscent(X, y, theta, alpha[i], epoch)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(epoch), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()
