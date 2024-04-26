import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, BatchNormalization, Dropout
import math

def get_input():
    try:
        ticker = input("Please enter a ticker symbol: ")
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey=Y8ZR7HEZ95DQQ9QH"
        r = requests.get(url)
        d = r.json()
        d1 = d['Time Series (Daily)']
        return ticker, d1
    except:
        print("Ticker not valid")


def preprocess(d1):
    # url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey=Y8ZR7HEZ95DQQ9QH"
    # r = requests.get(url)
    # d = r.json()
    # d1 = d['Time Series (Daily)']
    sorted_data = sorted(d1.items())  # ascending order
    last_date = [sorted_data[-1][0]]  # retrieves the last date
    data = []
    for index, (date, values) in enumerate(sorted_data):
        close = float(values['4. close'])
        if close == 0:
            data.append(data[index - 1])  # imputes missing value with last known price
        data.append(close)
    data = np.log(np.array(data) + 1)  # log transform
    scaled = [((i - min(data)) / (max(data) - min(data))) for i in data]  # min max scaling
    X = []
    y = []
    for i in range(60,len(scaled)):
      X.append(scaled[i-60:i])
      y.append(scaled[i])
    return X, y, max(data), min(data), last_date, scaled

def model(X,y):
    network = Sequential()
    network.add(LSTM(50, return_sequences=True, input_shape=(np.shape(X)[1], 1)))
    network.add(LSTM(50))
    network.add(Dropout(0.2))
    network.add(BatchNormalization())
    network.add(Dense(1))
    network.compile(loss='mean_squared_error', optimizer='adam')
    network.fit(X, y, epochs=2, batch_size=32)
    return network

def inverse_predict(model,last60,max,min):
  y_pred = model.predict(np.expand_dims(last60, axis=0))
  inverse_scaled = (y_pred[0][0]) * (max - min) + min #inverse min max scaling
  prediction = math.e ** (inverse_scaled) - 1 #inverse log transform
  prices_scale = np.array(last60) * (max - min) + min #inverse min max scaling
  prices_log = math.e ** (prices_scale) - 1 #inverse log transform
  prices = np.append(prices_log,prediction)
  return prediction, prices






