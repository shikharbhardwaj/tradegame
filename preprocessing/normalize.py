import sys
import numpy as np
import pandas as pd

# We need to normalize the thing.

if len(sys.argv) < 3:
    print("Usage python normalize.py [file_name] [output_name]")
    quit()

data_df = pd.read_csv(sys.argv[1])
data_df = data_df.drop(['tick', 'close'], axis = 1)

# We do a running Z normalization of input column

print(data_df.shape)

period = 96

mean_df = data_df.rolling(period).mean().dropna()
var_df = data_df.rolling(period).std().dropna()

# No normalization for periodic data
mean_df['min'] = 0
mean_df['hr'] = 0
mean_df['day_of_week'] = 0
mean_df['month'] = 0

var_df['min'] = 1
var_df['hr'] = 1
var_df['day_of_week'] = 1
var_df['month'] = 1

last_mean = mean_df.tail(1)
last_var = var_df.tail(1)

mean_pad = np.repeat(last_mean.values, period - 1, axis=0)
mean_pad = pd.DataFrame(data=mean_pad, columns=data_df.columns)
mean_df = mean_df.append(mean_pad)

var_pad = np.repeat(last_var.values, period - 1, axis=0)
var_pad = pd.DataFrame(data=var_pad, columns=data_df.columns)
var_df = var_df.append(var_pad)

print(mean_df.shape)

data_df -= mean_df
data_df /= var_df

data_df.to_csv(sys.argv[2], index=False)
