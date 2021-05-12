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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime 
import matplotlib.pyplot as plt
from utils import split_sequences, split_sequences_outputs, timesplit


df = pd.read_csv('us-counties.csv')

pd.set_option('display.max_rows', df.shape[0]+1)
cuya_data = df[(df.county == 'Cuyahoga') & (df.state == 'Ohio')]
frank_data = df[(df.county == 'Franklin') & (df.state == 'Ohio')]
trumbull_data = df[(df.county == 'Trumbull') & (df.state == 'Ohio')]
shortest_data = min(cuya_data, frank_data, trumbull_data, key=len)
# cuya_data = cuya_data[cuya_data.date.isin(shortest_data.date)]
# frank_data = frank_data[frank_data.date.isin(shortest_data.date)]
# trumbull_data = trumbull_data[trumbull_data.date.isin(shortest_data.date)]
data = pd.DataFrame()
data["cases"] = cuya_data["cases"].values
data["deaths"] = cuya_data["deaths"].values
data["date"] = pd.to_datetime(cuya_data["date"].values, format="%Y-%m-%d")

mobility = pd.read_csv('cuyahoga-mobility.csv')
mobility["date"] = pd.to_datetime(mobility["date"], format="%m/%d/%Y")
mobility = mobility[mobility.date.isin(cuya_data.date)]
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

n_steps = 3

trainX, trainY = split_sequences_outputs(train_data, n_steps, [0, 1, 2 ,3, 4])
testX, testY = split_sequences_outputs(np_data, n_steps, [0, 1, 2, 3, 4])
testX_post, testY_post = split_sequences_outputs(test_data, n_steps, [0, 1, 2, 3, 4])
dates, test_dates = split_sequences(data["date"].to_numpy().reshape(-1, 1), n_steps)
dates_post, test_dates_post = split_sequences(data.loc[test_mask, "date"].to_numpy().reshape(-1, 1), n_steps)
dates_pre, test_dates_pre = split_sequences(data.loc[train_mask, "date"].to_numpy().reshape(-1, 1), n_steps)


def lstm_model(n_steps, inputs, outputs):
	model = Sequential()
	model.add(LSTM(100, activation='relu', return_sequences=False, input_shape=(n_steps, len(inputs))))
	model.add(Dense(len(outputs)))
	model.compile(optimizer='adam', loss='mae', metrics=["mape", "mse"])
	return model

model = lstm_model(n_steps, inputs, outputs)
model.summary()
model.fit(trainX, trainY, nb_epoch=300, batch_size=50, verbose=2, validation_split=0.3)
predY = model.predict(trainX, verbose=0)
predY_post = model.predict(testX_post, verbose=0)
#results = model.evaluate(testX_post, testY_post, batch_size=50)
print("Cases MSE: ", mean_squared_error(testY_post[:, 0], predY_post[:, 0]))
print("Cases MAE: ", mean_absolute_error(testY_post[:, 0], predY_post[:, 0]))
print("Deaths MSE: ", mean_squared_error(testY_post[:, 1], predY_post[:, 1]))
print("Deaths MAE: ", mean_absolute_error(testY_post[:, 1], predY_post[:, 1]))
print("Test loss: ", )

def get_forecast(model, starting_data, length):
	curr_data = starting_data
	outputs = []
	for _ in range(length):
		input_data = np.array([curr_data])
		predY = model.predict(input_data)[0]
		outputs.append(predY)
		# predY = [[1,2,3,4]]
		# curr_data = [[1,2,3,4], [1,2,3,4]]
		new_data = np.delete(curr_data, 0, axis=0)
		new_data = np.vstack([new_data, predY])
		curr_data = new_data
	return np.array(outputs)

RANGE = 14

future_Y = get_forecast(model, testX_post[-1, :], RANGE)

testY = scaler.inverse_transform(testY)
predY = scaler.inverse_transform(predY)
testY_post = scaler.inverse_transform(testY_post)
predY_post = scaler.inverse_transform(predY_post)
future_Y_scaled = scaler.inverse_transform(future_Y)

new = pd.date_range(start=test_dates_post[-1, 0], periods=RANGE+1)
date_df = pd.DataFrame(new, columns=['date'])

plt.figure(1)
plt.plot(test_dates_post, testY_post[:, 0], label="Real cases")
#plt.plot(test_dates_pre, predY[:, 0], label="Pred cases")
plt.plot(test_dates_post, predY_post[:, 0], label="Pred cases on unseen data")
plt.plot(date_df['date'].loc[1:], future_Y_scaled[:, 0], label="Future predictions")
plt.title("Prediction vs Result")
plt.legend()
plt.figure(2)
# plt.plot(test_dates, testY[:, 1], label="Real deaths")
# plt.plot(test_dates_pre, predY[:, 1], label="Pred deaths")
# plt.plot(test_dates_post, predY_post[:, 1], label="Pred cases on unseen data")
plt.plot(test_dates_post, testY_post[:, 1], label="Real deaths")
plt.plot(test_dates_post, predY_post[:, 1], label="Pred deaths on unseen data")
plt.plot(date_df['date'].loc[1:], future_Y_scaled[:, 1], label="Future predictions")
plt.legend()
plt.title("Prediction vs Result")
plt.show()
