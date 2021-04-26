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
train_mask = data['date'] < pd.Timestamp(2020, 12, 31)
test_mask = data['date'] >= pd.Timestamp(2021, 1, 1)
train_data = data.loc[train_mask, inputs]
test_data = data.loc[test_mask, inputs]

n_steps = 3

trainX, trainY = split_sequences(train_data.to_numpy(), n_steps)
testX, testY = split_sequences(data[inputs].to_numpy(), n_steps)
testX_post, testY_post = split_sequences(test_data.to_numpy(), n_steps)
dates, test_dates = split_sequences(data["date"].to_numpy().reshape(-1, 1), n_steps)

def lstm_model(n_steps, inputs):
	model = Sequential()
	model.add(LSTM(32, activation='relu', return_sequences=False, input_shape=(n_steps, len(inputs))))
	model.add(Dense(len(inputs)))
	model.compile(optimizer='adam', loss='mae')
	return model

model = lstm_model(n_steps, inputs)
model.fit(trainX, trainY, nb_epoch=200, batch_size=50, verbose=2, validation_split=0.3)
predY = model.predict(testX, verbose=0)
results = model.evaluate(testX_post, testY_post, batch_size=50)

print("Test loss: ", results)

plt.figure(1)
plt.plot(test_dates, testY[:, 0], label="Real cases")
plt.plot(test_dates, predY[:, 0], label="Pred cases")
plt.legend()
plt.title("Prediction vs Result")
plt.figure(2)
plt.plot(test_dates, testY[:, 1], label="Real deaths")
plt.plot(test_dates, predY[:, 1], label="Pred deaths")
plt.legend()
plt.title("Prediction vs Result")
plt.show()
