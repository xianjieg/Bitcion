#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import datetime
import pytz
import statsmodels.api as sm  #用于单位根检验
import matplotlib.pyplot as plt  #画图
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA  #ARIMA模型
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[14]:


#转换为本地时间戳
def dateparse (time_in_secs):
    return pytz.utc.localize(datetime.datetime.fromtimestamp(float(time_in_secs)))
#数据导入
bitcoin = pd.read_csv(r"C:\Users\xianj\Desktop\bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv", parse_dates=[0], date_parser=dateparse)

#数据整理，按天合并，并对空缺值补前一个非空值
bitcoin['Timestamp'] = bitcoin['Timestamp'].dt.tz_localize(None)
bitcoin = bitcoin.groupby([pd.Grouper(key='Timestamp', freq='W-MON')]).first().reset_index()
bitcoin = bitcoin.set_index('Timestamp')
bitcoin = bitcoin[['Close']]
bitcoin['Close'] = bitcoin['Close'].fillna(method='ffill')

bitcoin.index = pd.to_datetime(bitcoin.index)
#print(bitcoin.head(20))

#拆分训练集和测试集，训练集时间为2018年12月17日至2020年3月31日
#测试集时间为2020年4月1日至4月20日
splitdate = '2018-12-17'
predate = '2020-04-01'
stop = '2020-04-21'
bitcoin_train = bitcoin.loc[bitcoin.index > splitdate]
bitcoin_train = bitcoin_train.loc[bitcoin_train.index <= predate]
bitcoin_test = bitcoin.loc[bitcoin.index < stop & bitcoin.index > predate]


# In[10]:


print(bitcoin.index)


# In[11]:


print(bitcoin_test)


# In[4]:


plt.figure(figsize=(12,5))
plt.plot(bitcoin_train)
plt.legend(bbox_to_anchor=(1.25, 0.5))
plt.title('Bitcoin Close')
sns.despine()

#单位根的检验成果
print(sm.tsa.stattools.adfuller(bitcoin_train))


# In[5]:


bitcoin_diff = bitcoin_train.diff(1)
bitcoin_diff = bitcoin_diff.dropna()
plt.plot(bitcoin_diff, label='first order difference')
plt.legend()
plt.title('first order difference')
plt.show()


# # ACF and PACF

# In[6]:


acf = plot_acf(bitcoin_diff, lags=30)
plt.title('ACF')
acf.show()
#绘制pacf图像
pacf = plot_pacf(bitcoin_diff, lags=30)
plt.title('PACF')
pacf.show()
plt.show()


# the figure show it is relatively stable, so the value of d is 1, and it can be determined from the ACF and PACF diagrams that when the values of p and q are both 1, it is guaranteed that 95% of the data basically fall within the confidence interval.

# In[7]:


model = ARIMA(bitcoin_train, order=(1,1,1))
result = model.fit()
print(result.summary()) #对模型进行分析
print(result.conf_int()) #查看每个系数的置信区间


# In[12]:


pred = result.predict('2020-04-02','2020-05-02', dynamic=True, typ='levels')
plt.figure(figsize=(12,5))
plt.title('Bitcoin Close Predict VS True value')
plt.plot(pred, label='predict value')
plt.plot(bitcoin_test, label='true value')
plt.legend()
plt.show()


# In[9]:


print(bitcoin_test)


# In[12]:


from sklearn.metrics import mean_squared_error
#计算误差
PreSquE = mean_squared_error(y_true=bitcoin_test.values, y_pred=pred.values)
PreAbE = mean_absolute_error(y_true=bitcoin_test.values, y_pred=pred.values)
print(PreSquE, PreAbE)


# In[13]:


print(bitcoin_test.values)


# In[14]:


print(pred.values)


# In[ ]:




