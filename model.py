import numpy as np
import scipy
import importlib 
import pandas as pd 
location_data = importlib.import_module("location-data")

df = pd.read_csv('us-counties.csv')
pd.set_option('display.max_rows', df.shape[0]+1)
data = df[(df.county == 'Cuyahoga') & (df.state == 'Ohio')]
print(location_data.timesplit(data, 1, ["cases", "deaths"]))


