import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##计算误差函数
def ComputeCost(X,y,theta):
    inner = np.power((np.dot(X,theta.T) - y),2)     #数列中各元素平方
    return np.sum(inner) / (2 * len(X))

##梯度下降
def GradientDescent(X,y,theta,alpha,epoch):
    temp = np.zeros(theta.shape)
    cost = np.zeros(epoch)
    m = X.shape[0]
    for i in range(epoch):
        temp = theta - np.dot((alpha / m) * (np.dot(X,theta.T) - y).T,X)
        theta = temp
        cost[i] = ComputeCost(X,y,theta)
    return theta,cost

##正则方程
def NormalEqn(X,y):
    theta = np.linalg.inv(X.T@X)@X.T@y
    return theta

##读取数据，并且查看
path = "ex1data1.txt"
data = pd.read_csv(path,header = None,names = ['Population','Profit']) #header为行名，names为列名
data.head()                 #head是读取前几位数据
data.describe()             #describe是对此数据集求均值方差等操作

data.plot(kind = 'scatter',x = 'Population',y = 'Profit',figsize = (8,5))  #绘制数据
plt.show()
data.insert(0,'Ones',1)     #第一个数字表示位置（0表示第一位），Ones是名称，1为插入的内容

cols = data.shape[1]        #获取列数(0 1)表示维度
X = data.iloc[:,0:cols-1]   #调用时需用values
y = data.iloc[:,cols-1:cols]

X = np.array(X.values)
y = np.array(y.values)

##赋初值
theta = np.array([[0,0]])
alpha = 0.01
epoch = 1000

##最终计算theta
final_theta,cost = GradientDescent(X,y,theta,alpha,epoch)

##绘制图像
x = np.linspace(data.Population.min(),data.Population.max(),100)    #确定x轴
f = final_theta[0,0] + (final_theta[0,1] * x)                       #确定y轴
fig,ax = plt.subplots(figsize = (6,4))                              #确定图像大小
ax.plot(x,f,'r',label = 'Prediction')
ax.scatter(data.Population,data.Profit,label = 'Train Data')
ax.legend(loc = 2)  #表示在左上角
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()


##从sklearn中调取工具包
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X,y)
x = np.array(X[:, 1])
f = model.predict(X).flatten()

##画图
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()
final_theta2=NormalEqn(X, y)


