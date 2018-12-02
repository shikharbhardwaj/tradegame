#!/usr/bin/env python3

import sys
import os
import pandas as pd

# Take file name from the command line.
if len(sys.argv) < 2:
    sys.exit("No filename")

fpath = sys.argv[1]
output_dir = 'sampled_data' if len(sys.argv) < 3 else sys.argv[2]

data_df = pd.read_csv(fpath, header=None)

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
sample_offset = '15T'

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
base = os.path.basename(fpath)
save_name = os.path.splitext(base)[0] + '_' + sample_offset + '.csv'
save_path = os.path.abspath(os.path.join(output_dir, save_name))
resampled_data.to_csv(save_path)
