import pandas as pd

driving_data = pd.read_csv("cuyahoga-driving-apple.txt", sep='\t').transpose()
transit_data = pd.read_csv("cuyahoga-transit-apple.txt", sep='\t').transpose()
walking_data = pd.read_csv("cuyahoga-walking-apple.txt", sep='\t').transpose()
pd.set_option('display.max_rows', driving_data.shape[0]+1)
data = pd.DataFrame()
data = pd.concat([driving_data, transit_data, walking_data], axis=1, ignore_index=True)
data.columns = ["driving-mobility", "transit-mobility", "walking-mobility"]
data["date"] = driving_data.index
data.index = list(range(len(data)))
data.to_csv("cuyahoga-mobility.csv")