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
import datetime 
import matplotlib.pyplot as plt
from utils import split_sequences, split_sequences_multi, timesplit

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

n_steps = 10
n_steps_out = 7

trainX, trainY = split_sequences_multi(train_data.to_numpy(), n_steps, n_steps_out)
testX, testY = split_sequences_multi(test_data.to_numpy(), n_steps, n_steps_out)
_, testY_flatten = split_sequences(test_data.to_numpy(), n_steps)
_, test_dates_multi = split_sequences_multi(data.loc[test_mask, "date"].to_numpy().reshape(-1, 1), n_steps, n_steps_out)
_, test_dates = split_sequences(data.loc[test_mask, "date"].to_numpy().reshape(-1, 1), n_steps)


print(trainX.shape, trainY.shape)

def lstm_model(n_steps, n_steps_out, inputs):
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_steps, len(inputs))))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(len(inputs))))
    model.compile(optimizer='adam', loss='mae')
    return model

model = lstm_model(n_steps, n_steps_out, inputs)
model.fit(trainX, trainY, nb_epoch=200, batch_size=50, verbose=2, validation_split=0.3)
predY = model.predict(testX, verbose=0)
results = model.evaluate(testX, testY, batch_size=50)
print("test loss:", results)



# For effective vizualiation, we are only showing test days since the overlapping future prediction will make the real data blurry
plt.figure(1)
plt.plot(test_dates, testY_flatten[:, 0])
for i in range(predY.shape[0]):
    plt.plot(test_dates_multi[i, :], predY[i, :, 0], color="orange")
plt.legend(["Real cases", "Pred cases"])
plt.title("Prediction vs Result multi-timestep")
plt.figure(2)
plt.plot(test_dates, testY_flatten[:, 1])
for i in range(predY.shape[0]):
    plt.plot(test_dates_multi[i, :], predY[i, :, 1], color="orange")
plt.legend(["Real deaths", "Pred deaths"])
plt.title("Prediction vs Result multi-timestep")
plt.show()
