import numpy as np
import scipy
import importlib 
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.losses import MAPE
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime 
import matplotlib.pyplot as plt
from utils import split_sequences, timesplit


df = pd.read_csv('us-counties.csv')
pd.set_option('display.max_rows', df.shape[0]+1)
data = df[(df.county == 'Cuyahoga') & (df.state == 'Ohio')]
data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
timesplit_data, outputs, inputs = timesplit(data, 2, ["cases", "deaths"])

    
inputs = ["cases", "deaths"]
for var in inputs:
    data[var] = data[var].diff()
    data[var] = data[var].dropna()
    data[var] = data[var].rolling(window=5).mean()

data = data.dropna()
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

n_steps = 3

trainX, trainY = split_sequences(train_data, n_steps)
testX, testY = split_sequences(np_data, n_steps)
testX_post, testY_post = split_sequences(test_data, n_steps)
dates, test_dates = split_sequences(data["date"].to_numpy().reshape(-1, 1), n_steps)
dates_post, test_dates_post = split_sequences(data.loc[test_mask, "date"].to_numpy().reshape(-1, 1), n_steps)
dates_pre, test_dates_pre = split_sequences(data.loc[train_mask, "date"].to_numpy().reshape(-1, 1), n_steps)

def lstm_model(n_steps, inputs):
	model = Sequential()
	model.add(LSTM(32, activation='relu', return_sequences=False, input_shape=(n_steps, len(inputs))))
	model.add(Dense(len(inputs)))
	model.compile(optimizer='adam', loss='mae', metrics=["mape", "mse"])
	return model

model = lstm_model(n_steps, inputs)
model.summary()
model.fit(trainX, trainY, nb_epoch=300, batch_size=50, verbose=2, validation_split=0.3)
predY = model.predict(trainX, verbose=0)
predY_post = model.predict(testX_post, verbose=0)

print("Cases MSE: ", mean_squared_error(testY_post[:, 0], predY_post[:, 0]))
print("Cases MAE: ", mean_absolute_error(testY_post[:, 0], predY_post[:, 0]))
print("Deaths MSE: ", mean_squared_error(testY_post[:, 1], predY_post[:, 1]))
print("Deaths MAE: ", mean_absolute_error(testY_post[:, 1], predY_post[:, 1]))
print("Test loss: ", )

testY = scaler.inverse_transform(testY)
predY = scaler.inverse_transform(predY)
predY_post = scaler.inverse_transform(predY_post)

plt.figure(1)
plt.plot(test_dates, testY[:, 0], label="Real cases")
plt.plot(test_dates_pre, predY[:, 0], label="Pred cases")
plt.plot(test_dates_post, predY_post[:, 0], label="Pred cases on unseen data")
plt.title("Prediction vs Result")
plt.legend()
plt.figure(2)
plt.plot(test_dates, testY[:, 1], label="Real deaths")
plt.plot(test_dates_pre, predY[:, 1], label="Pred deaths")
plt.plot(test_dates_post, predY_post[:, 1], label="Pred cases on unseen data")
plt.legend()
plt.title("Prediction vs Result")
plt.show()
