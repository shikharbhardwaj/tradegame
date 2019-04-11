# Combine monthly data to a single dataframe and save to disk.
import sys
from os import path
import pandas as pd
import platform

if len(sys.argv) != 3:
    print("python collate.py [pair] [year]")
    sys.exit(0)

pair = sys.argv[1]
year = sys.argv[2]

plat = platform.system()
base = 'D:'
if plat != 'Windows':
    base = 'data'

year_data = []

for i in range(1, 13):
    month = str(i).zfill(2)
    data_file = f'{pair}-{year}-{month}_15T.csv'
    data_location = path.join(base, 'tradegame_data', 'sampled_data_15T', data_file)
    df = pd.read_csv(data_location)
    year_data.append(df)

year_df = pd.concat(year_data)

file_name = f"{pair}_{year}.csv"
data_location = path.join(base, 'tradegame_data', 'sampled_data_15T', file_name[0:file_name.find('.')], file_name)

year_df.to_csv(data_location, index=False)

