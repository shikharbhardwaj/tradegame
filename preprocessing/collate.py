# Combine monthly data to a single dataframe and save to disk.
import sys
from os import path
import pandas as pd
import platform

plat = platform.system()
base = 'D:'
if plat != 'Windows':
    base = 'data'

DATA_PATH = path.join(base, 'tradegame_data', 'sampled_data_15T')

def collate(pair, year):
    year_data = []

    for i in range(1, 13):
        month = str(i).zfill(2)
        data_file = f'{pair}-{year}-{month}_15T.csv'
        data_location = path.join(DATA_PATH, data_file)
        df = pd.read_csv(data_location)
        year_data.append(df)

    year_df = pd.concat(year_data)

    # file_name = f"{pair}_{year}.csv"
    # data_location = path.join(base, 'tradegame_data', 'sampled_data_15T', 'yearly', file_name)

    # year_df.to_csv(data_location, index=False)

    return year_df

