import datetime
import pandas_datareader.data as web
import math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

'''日期'''
start = datetime.datetime(2018, 4, 1) # or start = '1/1/2016'
end = datetime.date.today()
print("蓝线是股价预测")
stock = str(input("输入股票代号:"))
#df = web.DataReader('AAPL', 'yahoo', start, end)
df = web.DataReader(stock, 'yahoo', start, end)
#print(df.keys())
#print(df.head())

# 定义预测列变量，它存放研究对象的标签名
'''预测范围'''
forecast_col = 'Close'
forecast_out = int(math.ceil(0.03*len(df)))
#print(len(df))
df["Change_pct"] = (df['Open']-df['Close'])/df['Open']*100.0
df['HL_pct'] = (df['High']-df['Low'])/df['Low']*100.0


df = df[['Close', 'HL_pct', 'Change_pct', 'Volume']]
df.fillna(-999999, inplace=True)


# 用label代表该字段，是预测结果
# 通过让与Close列的数据往前移动1%行来表示
df['label'] = df[forecast_col].shift(-forecast_out)
#print(df.head(10))
#print(forecast_out)

#最后生成真正在模型中使用的数据X和y和预测时用到的数据数据X_lately
X = np.array(df.drop(['label'],1))#删除label
#print(X[0:10])

# 使用sklearn.preprocessing.scale()函数，可以直接将给定数据进行标准化
X = preprocessing.scale(X)
#print(len(X))
# 上面生成label列时留下的最后1%行的数据，这些行并没有label数据，因此我们可以拿他们作为预测时用到的输入数据
X_lately = X[-forecast_out:] #最后forecast行 最后6hang
X = X[:-forecast_out] #一直到最后6行

#print(len(X_lately))
# 抛弃label列中为空的那些行
df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
# 生成scikit-learn的线性回归对象
clf = LinearRegression(n_jobs=-1)
# 开始训练
clf.fit(X_train, y_train)
# 用测试数据评估准确性
accuracy = clf.score(X_test, y_test)
# 进行预测
forecast_set = clf.predict(X_lately)

print(forecast_set,"\n", accuracy)


#画图
style.use("ggplot")
one_day = 86400
# 在df中新建Forecast列，用于存放预测结果的数据
df['Forecast'] = np.nan
# 取df最后一行的时间索引
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
next_unix = last_unix + one_day

# 遍历预测结果，用它往df追加行
# 这些行除了Forecast字段，其他都设为np.nan
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    # [np.nan for _ in range(len(df.columns) - 1)]生成不包含Forecast字段的列表
    # 而[i]是只包含Forecast值的列表
    # 上述两个列表拼接在一起就组成了新行，按日期追加到df的下面
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

# 开始绘图
df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()