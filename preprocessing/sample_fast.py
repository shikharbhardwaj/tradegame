#!/usr/bin/env python3

import pandas as pd
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

resampled_data = data_df.resample('15T', on='tick').first()
resampled_data.columns = ['tick', 'bid', 'open']
del resampled_data['bid']

resampled_data['high'] = data_df.resample('15T', on='tick').max()['ask']
resampled_data['low'] = data_df.resample('15T', on='tick').min()['ask']
resampled_data['close'] = data_df.resample('15T', on='tick').last()['ask']
resampled_data['volume'] = data_df.resample('15T', on='tick').count()['ask']


# Save processed data to disk.
save_name = fname[0:fname.rfind('.')] + '_features_fast.csv'
# Create a dataframe.
print("Writing")
resampled_data.to_csv(save_name, index=False)
