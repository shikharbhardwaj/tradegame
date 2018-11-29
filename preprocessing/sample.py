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

last_time = data_df['tick'][0]
cur_price = data_df['ask'][0]

# Each 15 minute interval is represented by the following variables.
cur_interval = {'date': last_time,
                'open': cur_price,
                'high': cur_price,
                'low': cur_price,
                'close': cur_price,
                'volume': 0}

last_time = last_time.timestamp()
intervals = []

# Sample the intervals from the raw data.
for row in data_df.itertuples():
    cur_time = row.tick
    cur_price = row.ask
    
    # The sampling interval is 900 seconds.
    if row.tick.timestamp() - last_time > 900:
        # Append this to the list of intervals.
        intervals.append(cur_interval.copy())
        
        cur_interval['date'] = row.tick
        cur_interval['open'] = row.ask
        cur_interval['high'] = row.ask
        cur_interval['low'] = row.ask
        cur_interval['close'] = row.ask
        cur_interval['volume'] = 1
        last_time = row.tick.timestamp()
    else:
        cur_interval['close'] = row.ask
        cur_interval['high'] = max(row.ask, cur_interval['high'])
        cur_interval['low'] = min(row.ask, cur_interval['low'])
        cur_interval['volume'] += 1
        
# The last remaining (incomplete) interval.
intervals.append(cur_interval.copy())

# Save processed data to disk.
save_name = fname[0:fname.rfind('.')] + '_features.csv'
# Create a dataframe.
print("Creating")
processed_data = pd.DataFrame(intervals)
print("Writing")
processed_data.to_csv(save_name, index=False)
