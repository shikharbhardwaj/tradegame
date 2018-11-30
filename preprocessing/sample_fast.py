#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys

# Take file name from the command line.

if len(sys.argv) < 2:
    print("No filename.")
    quit()

fname = './AUDJPY-2012-01.csv'

data_df = pd.read_csv(fname, header=None)

# Let the dataframe about the columns of the CSV
data_df.columns = ['name', 'tick', 'bid', 'ask']

# Let the dataframe know the timestamp column
data_df['tick'] = pd.to_datetime(data_df['tick'])
del data_df['name']
# Index on the timestamp
data_df = data_df.set_index('tick')


# We resample in this offset.
# Docs:
# http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
sample_offset = '24H'

resampled_data = data_df.resample(sample_offset).first()
del resampled_data['bid']
resampled_data.columns = ['open']

resampled_data['high'] = data_df.resample(sample_offset).max()['ask']
resampled_data['low'] = data_df.resample(sample_offset).min()['ask']
resampled_data['close'] = data_df.resample(sample_offset).last()['ask']
resampled_data['volume'] = data_df.resample(sample_offset).count()['ask']

# We need to remove bad values.
resampled_data = resampled_data.dropna()

# Save processed data to disk.
save_name = fname[0:fname.rfind('.')] + '_daily_fast.csv'
print("Writing")
resampled_data.to_csv(save_name)
