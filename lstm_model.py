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
import datetime 
import matplotlib.pyplot as plt

location_data = importlib.import_module("location-data")

df = pd.read_csv('us-counties.csv')
pd.set_option('display.max_rows', df.shape[0]+1)
data = df[(df.county == 'Cuyahoga') & (df.state == 'Ohio')]
data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
timesplit_data, outputs, inputs = location_data.timesplit(data, 2, ["cases", "deaths"])


def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)
inputs = ["cases", "deaths"]
for var in inputs:
    data[var] = data[var].diff()
    data[var] = data[var].dropna()
    data[var] = data[var].rolling(window=5).mean()

data = data.dropna()
train_mask = data['date'] < pd.Timestamp(2020, 12, 31)
test_mask = data['date'] >= pd.Timestamp(2021, 1, 1)
train_data = data.loc[train_mask, inputs]
test_data = data.loc[test_mask, inputs]

n_steps = 3

trainX, trainY = split_sequences(train_data.to_numpy(), n_steps)
testX, testY = split_sequences(data[inputs].to_numpy(), n_steps)
dates, test_dates = split_sequences(data["date"].to_numpy().reshape(-1, 1), n_steps)

model = Sequential()
model.add(LSTM(32, activation='relu', return_sequences=True, input_shape=(n_steps, len(inputs))))
model.add(LSTM(32, activation='relu'))
model.add(Dense(len(inputs)))
model.compile(optimizer='adam', loss='mae')

model.fit(trainX, trainY, nb_epoch=200, batch_size=50, verbose=2, validation_split=0.3)
predY = model.predict(testX, verbose=0)
plt.subplot(1, 2, 1)
plt.plot(test_dates, testY[:, 0], label="Real cases")
plt.plot(test_dates, predY[:, 0], label="Pred cases")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(test_dates, testY[:, 1], label="Real deaths")
plt.plot(test_dates, predY[:, 1], label="Pred deaths")
plt.legend()
plt.title("Prediction vs Result")
plt.show()
