# 导入需要的模块
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# 读取特斯拉股票数据
df = pd.read_csv('TSLA.csv')
# 只保留收盘价这一列
df = df['Close']
# 将数据转换为 numpy 数组
data = df.values
# 定义训练集和测试集的大小
train_size = int(len(data) * 0.8)
test_size = len(data) - train_size
# 分割训练集和测试集
train_data = data[:train_size]
test_data = data[train_size:]
# 创建一个归一化对象
scaler = MinMaxScaler()
# 对训练集进行归一化
train_data = scaler.fit_transform(train_data.reshape(-1, 1))
# 对测试集进行归一化
test_data = scaler.transform(test_data.reshape(-1, 1))
# 定义一个函数，根据过去 n 天的数据来生成输入和输出
def create_dataset(data, n):
    x = []
    y = []
    for i in range(n, len(data)):
        x.append(data[i-n:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)
# 定义过去天数为 60
n = 60
# 创建训练集和测试集的输入和输出
x_train, y_train = create_dataset(train_data, n)
x_test, y_test = create_dataset(test_data, n)
# 调整输入的形状，以适应 LSTM 模型
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
# 创建一个 LSTM 模型
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(n, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
# 编译模型，使用均方误差作为损失函数，使用 Adam 优化器
model.compile(loss='mean_squared_error', optimizer='adam')
# 训练模型，使用 20 个周期，每个批次 32 个样本
model.fit(x_train, y_train, epochs=20, batch_size=32)
# 预测测试集的股价
y_pred = model.predict(x_test)
# 反归一化预测结果和真实结果
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
# 绘制预测结果和真实结果的对比图
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('TSLA Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
