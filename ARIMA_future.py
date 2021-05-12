from statsmodels.tsa.arima.model import ARIMA
from utils import *
import pandas as pd
import matplotlib.pyplot as plt

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

model = ARIMA(endog=data["deaths"], order=(2, 2, 2))
res = model.fit()
RANGE = 14
print(res.get_prediction(end=len(data) + RANGE).predicted_mean)
new = pd.date_range(data["date"].iloc[0], periods=len(data) + RANGE + 1)
df = pd.DataFrame(new, columns=['date'])
plt.figure(1)
plt.plot(data["date"], data["deaths"], label="Real deaths")
plt.plot(df["date"], res.get_prediction(end=len(data) + RANGE).predicted_mean, label="Pred deaths")
plt.legend()
plt.title("Prediction vs Result")
plt.show()