import pandas as pd

df = pd.read_csv('us-counties.csv')
pd.set_option('display.max_rows', df.shape[0]+1)
print(df[(df.county == 'Cuyahoga') & (df.state == 'Ohio')])