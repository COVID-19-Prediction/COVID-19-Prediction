from statsmodels.tsa.arima.model import ARIMA
from utils import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

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
scaler = MinMaxScaler(feature_range=(0, 1))
np_data = data.loc[:, ["cases", "deaths"]].to_numpy()
train_mask = data['date'] < pd.Timestamp(2020, 12, 31)
test_mask = data['date'] >= pd.Timestamp(2021, 1, 1)
np_data = data.loc[:, inputs].to_numpy()
np_data = scaler.fit_transform(np_data)
train_data = np_data[train_mask, :]
test_data = np_data[test_mask, :]
model = ARIMA(endog=train_data[:, 0], order=(2, 2, 2))
res = model.fit()
test_res = res.apply(test_data[:, 0])
print(data)
plt.figure(1)
plt.plot(data.loc[:, "date"], np_data[:, 0], label="Real deaths")
plt.plot(data.loc[train_mask, "date"], res.get_prediction().predicted_mean, label="Pred deaths")
plt.plot(data.loc[test_mask, "date"], test_res.get_prediction().predicted_mean, label="Pred deaths with new data")
plt.legend()
plt.title("Prediction vs Result")
plt.show()
print(mean_squared_error(test_data[:, 0], test_res.get_prediction().predicted_mean))
print(mean_absolute_error(test_data[:, 0], test_res.get_prediction().predicted_mean))