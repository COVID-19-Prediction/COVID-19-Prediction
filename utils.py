import numpy as np
import pandas as pd
from typing import List

def split_sequences_multi(sequences, before, after):
	X, y = [], []
	for i in range(len(sequences)):
		end = i + before + after
		if end > len(sequences)-1:
			break
		seq_x, seq_y = sequences[i:i+before, :], sequences[i+before:end, :]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def split_sequences(sequences, before):
	X, y = [], []
	for i in range(len(sequences)):
		end = i + before
		if end > len(sequences)-1:
			break
		seq_x, seq_y = sequences[i:end, :], sequences[end, :]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def timesplit(df: pd.DataFrame, t, vars: List[str]):
    assert t > 0, "Timeshift must be larger than 0"
    out_df = df.iloc[t:-2]
    plus_range = []
    minus_range = []
    for var in vars:
        out_df[f"{var}+1"] = df[var].shift(-1)
        plus_range.append(f"{var}+1")
        minus_range.append(var)
        for i in range(1, t+1):
            out_df[f"{var}-{i}"] = df[var].shift(i)
            minus_range.append(f"{var}-{i}")
    return out_df, plus_range, minus_range