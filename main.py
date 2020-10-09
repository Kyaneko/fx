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
from oandapyV20 import API
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments
from datetime import datetime

ACCOUNT_ID = "101-009-16693469-001"
ACCESS_TOKEN = "27c3a5e78406e721fc749f56a6155549-55e99c7ed9ab96eb9d256159b00aa10b"

api = API(access_token=ACCESS_TOKEN, environment="practice")

# +
# 30分足の価格データの取得
'''
    date：日時の始点を表すdatetime
    count：データの長さ
    pair：通貨ペア        
'''

def getDataFromOanda(date, count, pair="USD_JPY"):
    params = {
        "to": date.isoformat(),
        "count": count,
        "granularity": "M30",
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

getDataFromOanda(datetime(2008,1,17,22,3,0),10)

datetime.now()

datetime(2000,10,3,23,0,59)

aa=datetime(2008,1,17,22,3,0).isoformat()
print(aa)
params = {
    "to": aa,
    "count": 10,
    "granularity": "M30",
}
r = instruments.InstrumentsCandles(instrument="USD_JPY", params=params)
api.request(r)

datetime.datetime()
