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
from utils import split_sequences, split_sequences_outputs, timesplit


df = pd.read_csv('us-counties.csv')

pd.set_option('display.max_rows', df.shape[0]+1)
cuya_data = df[(df.county == 'Cuyahoga') & (df.state == 'Ohio')]
frank_data = df[(df.county == 'Franklin') & (df.state == 'Ohio')]
trumbull_data = df[(df.county == 'Trumbull') & (df.state == 'Ohio')]
shortest_data = min(cuya_data, frank_data, trumbull_data, key=len)
cuya_data = cuya_data[cuya_data.date.isin(shortest_data.date)]
frank_data = frank_data[frank_data.date.isin(shortest_data.date)]
trumbull_data = trumbull_data[trumbull_data.date.isin(shortest_data.date)]
data = pd.DataFrame()
data["cases"] = cuya_data["cases"].values
data["deaths"] = cuya_data["deaths"].values
data["F-deaths"] = frank_data["deaths"].values
data["F-cases"] = frank_data["cases"].values
data["T-deaths"] = trumbull_data["deaths"].values
data["T-cases"] = trumbull_data["cases"].values
data["date"] = pd.to_datetime(cuya_data["date"].values, format="%Y-%m-%d")

mobility = pd.read_csv('cuyahoga-mobility.csv')
mobility["date"] = pd.to_datetime(mobility["date"], format="%m/%d/%Y")
mobility = mobility[mobility.date.isin(shortest_data.date)]
data["transit-mobility"] = mobility["transit-mobility"].values
data["driving-mobility"] = mobility["walking-mobility"].values
data["walking-mobility"] = mobility["walking-mobility"].values


inputs = ["cases", "deaths", "transit-mobility", "walking-mobility", "driving-mobility"]
outputs = ["cases", "deaths"]
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

trainX, trainY = split_sequences_outputs(train_data.to_numpy(), n_steps, [0, 1])
testX, testY = split_sequences_outputs(data[inputs].to_numpy(), n_steps, [0, 1])
testX_post, testY_post = split_sequences_outputs(test_data.to_numpy(), n_steps, [0, 1])
dates, test_dates = split_sequences(data["date"].to_numpy().reshape(-1, 1), n_steps)

def lstm_model(n_steps, inputs, outputs):
	model = Sequential()
	model.add(LSTM(50, activation='relu', return_sequences=False, input_shape=(n_steps, len(inputs))))
	model.add(Dense(len(outputs)))
	model.compile(optimizer='adam', loss='mae')
	return model

model = lstm_model(n_steps, inputs, outputs)
model.fit(trainX, trainY, nb_epoch=100, batch_size=50, verbose=2, validation_split=0.3)
predY = model.predict(testX, verbose=0)
results = model.evaluate(testX_post, testY_post, batch_size=50)

print("Test loss: ", results)

plt.figure(1)
plt.plot(test_dates, testY[:, 0], label="Real cases")
plt.plot(test_dates, predY[:, 0], label="Pred cases")
plt.title("Prediction vs Result")
plt.legend()
plt.figure(2)
plt.plot(test_dates, testY[:, 1], label="Real deaths")
plt.plot(test_dates, predY[:, 1], label="Pred deaths")
plt.legend()
plt.title("Prediction vs Result")
plt.show()
