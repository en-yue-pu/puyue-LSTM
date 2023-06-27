# puyue-LSTM
用 python 写一个预测股价的算法
使用一种叫做 长短期记忆网络（LSTM） 的人工神经网络来处理时间序列数据，如股票价格。
LSTM 可以记住过去的信息，并根据当前的输入来预测未来的输出

其中使用了 tensorflow 这个模块 必须要x86设备才能运行(mac不行)
python3.8(3.9和以后不能用)

从网上下载你感兴趣的股票的历史数据，如从 Yahoo 财经下载特斯拉（TSLA）的数据。
使用 pandas 库来读取和处理数据，提取收盘价作为特征和目标变量。
使用 sklearn 库中的 MinMaxScaler 类来对数据进行归一化，使其在 0 到 1 之间。
创建一个训练集和一个测试集，分别包含过去 60 天和未来一天的数据。
使用 keras 库中的 Sequential 类来创建一个 LSTM 模型，添加一些 LSTM 层，全连接层和 Dropout 层来防止过拟合。
使用 fit 方法来训练模型，使用 predict 方法来预测测试集中的股价。
使用 matplotlib 库来绘制预测结果和真实结果的对比图，评估模型的表现。

usage
pip install -r requirements.txt 
python main.py

<img width="637" alt="截屏2023-06-27 23 50 25" src="https://github.com/yue-pu/puyue-LSTM/assets/117700148/e1c4afc2-3d46-4257-8f74-09bab7569b30">
