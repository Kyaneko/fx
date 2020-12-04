# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import os, sys, glob, random, csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import datetime
from oandapyV20 import API
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments
import tensorflow as tf

ACCOUNT_ID = "101-009-16693469-001"
ACCESS_TOKEN = "27c3a5e78406e721fc749f56a6155549-55e99c7ed9ab96eb9d256159b00aa10b"

api = API(access_token=ACCESS_TOKEN, environment="practice")


# +
def to_iso(time):
    '''
        (input)
            time：datetime(年,月,日,時,分)の形式
        
        (output)
            OandaのAPIに入力するためのisoフォーマット
    '''
    return time.isoformat()+'.000000000Z'

def yesterday_date():
    '''
        (input)
            なし

        (output)
            前日0時を表すdatetime
    '''
    t = datetime.datetime.now()
    t -= datetime.timedelta(
        days = 1,
        hours = t.hour,
        minutes = t.minute,
        seconds = t.second+t.microsecond/(10**6)
    )
    return t

def getDataFromOanda(count, time, pair='USD_JPY', granularity='H1'):
    '''
        (input)
            time：日時の終点を表すdatetime
            count：データ数
            pair：通貨ペア
            granularity：価格データの取得間隔
        
        (output)
            価格データのリスト
    ''' 
    params = {
        'to': to_iso(time),
        'count': count,
        'granularity': granularity,
    }
    r = instruments.InstrumentsCandles(instrument=pair, params=params)
    candles = api.request(r)
    res = []
    for c in candles['candles']:
        tmp = []
        for k in ['o', 'h', 'l', 'c']:
            tmp.append(float(c['mid'][k]))
        res.append(tmp)    
    return res


# -

def getDatasetFromOanda(count, timesteps, to=yesterday_date(), pair='USD_JPY', granularity='H1'):
    '''
        (input)
            count：データ数
            timesteps：予測に使う時系列数
            to：日時の終点を表すdatetime
            pair：通貨ペア
            granularity：価格データの取得間隔

        (output)
            入力・ラベルのndarray
    '''
    lis = getDataFromOanda(count+timesteps, to, pair, granularity)
    x = []
    y = []
    for i in range(count):
        x.append(lis[i:i+timesteps])
        y.append(lis[i+timesteps])
    return np.array(x), np.array(y)


x,y=getDatasetFromOanda(5, 3)
x

print(datetime.datetime.now())
print(yesterday_date())

aa=datetime(2020,12,4,9).isoformat()+'.000000000Z'
#aa=datetime.now().isoformat()+'.000000000Z'
print(aa)
params = {
    "to": aa,
    "count": 1000,
    "granularity": "H1",
}
r = instruments.InstrumentsCandles(instrument="USD_JPY", params=params)
c = api.request(r)
c

# 学習モデル
tf.__version__


# 学習モデル
class myModel(tf.keras.Model):
    
    def __init__(self):
        
        super(myModel, self).__init__()
        
        self.flatten0 = tf.keras.layers.Flatten()
        self.flatten1 = tf.keras.layers.Flatten()
        self.flatten2 = tf.keras.layers.Flatten()
        self.flatten3 = tf.keras.layers.Flatten()
        
        self.ave1 = tf.keras.layers.AveragePooling2D((1,4), strides=1)
        self.ave2 = tf.keras.layers.AveragePooling2D((5,1), strides=1)
        self.ave3 = tf.keras.layers.AveragePooling2D((25,1), strides=1)
        
        self.dense0 = tf.keras.layers.Dense(32)
        self.dense1 = tf.keras.layers.Dense(32)
        self.dense2 = tf.keras.layers.Dense(32)
        self.dense3 = tf.keras.layers.Dense(32)
        self.dense4 = tf.keras.layers.Dense(128)    
        self.dense5 = tf.keras.layers.Dense(100) 
                
        self.concate_x = tf.keras.layers.Concatenate()
        self.concate_y = tf.keras.layers.Concatenate(axis=1)
        
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.bn = tf.keras.layers.BatchNormalization()
        self.reshape = tf.keras.layers.Reshape((25,4,1))
        self.add = tf.keras.layers.Add()
        
    def call(self, x):
              
        x0 = x
        x0 = self.flatten0(x0)
        x0 = self.dense0(x0)
        
        x1 = x
        x1 = self.ave1(x1)
        x1 = self.flatten1(x1)
        x1 = self.dense1(x1)
        
        x2 = x
        x2 = self.ave2(x2)
        x2 = self.flatten2(x2)
        x2 = self.dense2(x2)
        
        x3 = x
        x3 = self.ave3(x3)
        x3 = self.flatten3(x3)
        x3 = self.dense3(x3)
        
        d = self.concate_x([x0,x1,x2,x3])
        d = self.dropout(d)
        d = self.bn(d)
        d = self.dense4(d)
        d = self.dense5(d)
        d = self.reshape(d)
        
        y = tf.expand_dims(x[:,-1], axis=1)
        y = self.concate_y([y]*25)
        
        y = self.add([y,d])
        
        return y


# +
model = tf.keras.models.Sequential([
    IdentityLayer(input_shape=(50,4,1)),
    myModel()
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
)
                             
model.summary()
# -

x = tf.random.normal((1,10,4,1))
#x = tf.keras.layers.Concatenate(axis=1)([x[:,-1]]*10)
y = np.expand_dims(x[:,-1],axis=1)

inputs = tf.random.normal([1, 4, 4])
gru = tf.keras.layers.GRU(4,return_sequences=True)
output = gru(inputs)
print(output, inputs)
