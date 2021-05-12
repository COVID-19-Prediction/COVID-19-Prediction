import numpy as np
import scipy
import importlib 
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Activation, RepeatVector
from tensorflow.keras.losses import MAPE
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import datetime 
import matplotlib.pyplot as plt
from utils import split_sequences, split_sequences_multi, split_sequences_multi_outputs, timesplit

df = pd.read_csv('us-counties.csv')
pd.set_option('display.max_rows', df.shape[0]+1)
data = df[(df.county == 'Cuyahoga') & (df.state == 'Ohio')]
data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
timesplit_data, outputs, inputs = timesplit(data, 2, ["cases", "deaths"])
mobility = pd.read_csv('cuyahoga-mobility.csv')
mobility["date"] = pd.to_datetime(mobility["date"], format="%m/%d/%Y")
mobility = mobility[mobility.date.isin(data.date)]
data["transit-mobility"] = mobility["transit-mobility"].values
data["driving-mobility"] = mobility["walking-mobility"].values
data["walking-mobility"] = mobility["walking-mobility"].values
    
inputs = ["cases", "deaths", "transit-mobility", "walking-mobility", "driving-mobility"]
outputs = ["cases", "deaths", "transit-mobility", "walking-mobility", "driving-mobility"]
for var in inputs:
    data[var] = data[var].diff()
    data[var] = data[var].dropna()
    data[var] = data[var].rolling(window=5).mean()

scaler = MinMaxScaler(feature_range=(0, 1))

data = data.dropna()
train_mask = data['date'] < pd.Timestamp(2020, 12, 31)
test_mask = data['date'] >= pd.Timestamp(2021, 1, 1)
train_data = data.loc[train_mask, inputs]
test_data = data.loc[test_mask, inputs]
np_data = data.loc[:, inputs].to_numpy()
np_data = scaler.fit_transform(np_data)
train_data = np_data[train_mask, :]
test_data = np_data[test_mask, :]

n_steps = 10
n_steps_out = 7

trainX, trainY = split_sequences_multi_outputs(train_data, n_steps, n_steps_out, [0, 1, 2, 3, 4])
testX, testY = split_sequences_multi_outputs(test_data, n_steps, n_steps_out, [0, 1, 2, 3, 4])
_, testY_flatten = split_sequences(test_data, n_steps)
_, test_dates_multi = split_sequences_multi(data.loc[test_mask, "date"].to_numpy().reshape(-1, 1), n_steps, n_steps_out)
_, test_dates = split_sequences(data.loc[test_mask, "date"].to_numpy().reshape(-1, 1), n_steps)


def lstm_model(n_steps, n_steps_out, inputs, outputs):
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_steps, len(inputs))))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(len(outputs))))
    model.compile(optimizer='adam', loss='mae', metrics=["mae", "mse"])
    return model
model = lstm_model(n_steps, n_steps_out, inputs, outputs)
model.fit(trainX, trainY, nb_epoch=200, batch_size=50, verbose=2, validation_split=0.3)
predY = model.predict(testX, verbose=0)
results = model.evaluate(testX, testY, batch_size=50)
print("test loss:", results)

# For effective vizualiation, we are only showing test days since the overlapping future prediction will make the real data blurry
plt.figure(1)
plt.plot(test_dates, scaler.inverse_transform(testY_flatten)[:, 0])
for i in range(predY.shape[0]):
    plt.plot(test_dates_multi[i, :], scaler.inverse_transform(predY[i])[:, 0], color="orange")
plt.legend(["Real cases", "Pred cases"])
plt.title("Prediction vs Result multi-timestep")
plt.figure(2)
plt.plot(test_dates, scaler.inverse_transform(testY_flatten)[:, 1])
for i in range(predY.shape[0]):
    plt.plot(test_dates_multi[i, :], scaler.inverse_transform(predY[i])[:, 1], color="orange")
plt.legend(["Real deaths", "Pred deaths"])
plt.title("Prediction vs Result multi-timestep")
plt.show()
