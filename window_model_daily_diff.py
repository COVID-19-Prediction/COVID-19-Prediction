import numpy as np
import scipy
import importlib 
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MAPE
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime 
import matplotlib.pyplot as plt

from utils import timesplit

df = pd.read_csv('us-counties.csv')
pd.set_option('display.max_rows', df.shape[0]+1)
data = df[(df.county == 'Cuyahoga') & (df.state == 'Ohio')]
data, outputs, inputs = timesplit(data, 2, ["cases", "deaths"])
data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
for var in inputs + outputs:
    data[var] = data[var].diff()
    data[var] = data[var].dropna()
    data[var] = data[var].rolling(window=5).mean()

data = data.dropna()

def window_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(13, input_dim=input_dim, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(output_dim))
    model.compile(loss="mae", optimizer='adam', metrics=["mape", "mse"])
    return model

print(inputs)
print(outputs)

train_mask = data['date'] < pd.Timestamp(2020, 12, 31)
test_mask = data['date'] >= pd.Timestamp(2021, 1, 1)
trainX = data.loc[train_mask, inputs]
trainY = data.loc[train_mask, outputs]
testX = data.loc[test_mask, inputs]
testY = data.loc[test_mask, outputs]
model = window_model(len(inputs), len(outputs))
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
X = scaler_X.fit_transform(data.loc[:, inputs])
Y = scaler_Y.fit_transform(data.loc[:, outputs])

trainX = X[train_mask, :]
trainY = Y[train_mask, :]
testX = X[test_mask, :]
testY = Y[test_mask, :]
model.summary()
model.fit(trainX, trainY, nb_epoch=1000, batch_size=50, verbose=2, validation_split=0.3)


result = model.evaluate(testX, testY)
print("MAE, MAPE, RMSE: ", result)


predY_post = model.predict(testX, verbose=0)
predY_pre = model.predict(trainX, verbose=0)

print("Cases MSE: ", mean_squared_error(testY[:, 0], predY_post[:, 0]))
print("Cases MAE: ", mean_absolute_error(testY[:, 0], predY_post[:, 0]))
print("Deaths MSE: ", mean_squared_error(testY[:, 1], predY_post[:, 1]))
print("Deaths MAE: ", mean_absolute_error(testY[:, 1], predY_post[:, 1]))



predY_post = scaler_Y.inverse_transform(predY_post)
predY_pre = scaler_Y.inverse_transform(predY_pre)


plt.figure(1)
plt.plot(data.loc[:, "date"], data["cases+1"], label="Real cases")
plt.plot(data.loc[train_mask, "date"], predY_pre[:, 0], label="Pred cases")
plt.plot(data.loc[test_mask, "date"], predY_post[:, 0], label="Pred cases unseen data")
plt.title("Prediction vs Result")
plt.legend()
plt.figure(2)
plt.plot(data.loc[:, "date"], data["deaths+1"], label="Real deaths")
plt.plot(data.loc[train_mask, "date"], predY_pre[:, 1], label="Pred deaths")
plt.plot(data.loc[test_mask, "date"], predY_post[:, 1], label="Pred deaths unseen data")
plt.legend()
plt.title("Prediction vs Result")
plt.show()

