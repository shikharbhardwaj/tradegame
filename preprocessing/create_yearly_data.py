import platform
from os import path
from tqdm import tqdm

from collate import collate

plat = platform.system()
base = 'D:'
if plat != 'Windows':
    base = 'data'

EXPORT_PATH = path.join(base, 'tradegame_data', 'sampled_data_15T', 'yearly')

pairs = ['AUDJPY', 'AUDNZD', 'AUDUSD', 'CADJPY', 'CHFJPY', 'EURCHF', 'EURGBP',
         'EURJPY', 'EURUSD', 'GBPJPY', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']


for year in range(2012, 2018):
    for pair in tqdm(pairs):
        year_df = collate(pair, year)

        file_name = f"{pair}_{year}.csv"
        data_location = path.join(base, 'tradegame_data', 'sampled_data_15T', 'yearly', file_name)

        year_df.to_csv(data_location, index=False)

    print()

