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
import datetime 
import matplotlib.pyplot as plt
from utils import timesplit

location_data = importlib.import_module("location-data")

df = pd.read_csv('us-counties.csv')
pd.set_option('display.max_rows', df.shape[0]+1)
data = df[(df.county == 'Cuyahoga') & (df.state == 'Ohio')]
data, outputs, inputs = timesplit(data, 2, ["cases", "deaths"])
data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
for var in inputs + outputs:
    data[var] = data[var].diff()


def window_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(13, input_dim=input_dim, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(output_dim, kernel_initializer='normal'))
    model.compile(loss="mean_squared_error", optimizer='adam', metrics=["mape", "accuracy"])
    return model

print(inputs)
print(outputs)

train_mask = data['date'] < pd.Timestamp(2020, 12, 31)
test_mask = data['date'] >= pd.Timestamp(2021, 1, 1)
trainX = data.loc[train_mask, inputs]
trainY = data.loc[train_mask, outputs]
model = window_model(len(inputs), len(outputs))
model.fit(trainX, trainY, nb_epoch=100, batch_size=50, verbose=2, validation_split=0.2)

testX = data.loc[:, inputs]
testY = data.loc[:, outputs]
dates = data["date"]

predY = model.predict(testX, verbose=0)
plt.figure(figsize=(9, 6))
plt.plot(dates, data["cases+1"], label="Real cases")
plt.plot(dates, data["deaths+1"], label="Real deaths")
plt.plot(dates, predY[:, 0], label="Pred cases")
plt.plot(dates, predY[:, 1], label="Pred deaths")
plt.legend()
plt.title("Prediction vs Result")
plt.show()

