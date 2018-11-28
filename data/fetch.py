#!/usr/bin/env python3

import calendar
import wget

pairs = ['AUDJPY', 'AUDNZD', 'AUDUSD', 'CADJPY', 'CHFJPY', 'EURCHF', 'EURGBP',
         'EURJPY', 'EURUSD', 'GBPJPY', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']

months = [calendar.month_name[i].upper() for i in range(1, 13)]

years = ['2012', '2013', '2014', '2015', '2016', '2017']


for year in years:
    for i, month in enumerate(months):
        for pair in pairs:
            month_idx = str(i + 1)
            if len(month_idx) == 1:
                month_idx = "0" + month_idx
            url = f'https://truefx.com/dev/data/{year}/{month}-{year}/{pair}-{year}-{month_idx}.zip'
            print(url)
