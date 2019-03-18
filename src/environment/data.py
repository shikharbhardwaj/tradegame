"""Data streaming functions for getting sampled currency pair data.
This module is responsible for data preprocessing:
imputation, normalization and subsetting.
"""

import pandas as pd
from os import path
import numpy as np

class BatchIterator:
    def __init__(self, location, currencyPairs, beginYear, endYear, period = 96, preprocess = True):
        """Initialize the batch generator

        Arguments:
            location {path} -- Path to the data folder
            currencyPairs {list} -- List of strings with the pair names
            beginYear {int} -- beginning of the dataset
            endYear {int} -- past the ending year of the dataset
            period {int} -- Period for rolling window normalization
            preprocess {bool} -- Preprocess tick data or not
        """
        self.location = location
        self.currency_pairs = currencyPairs
        self.begin_year = beginYear
        self.end_year = endYear
        self.period = period
        self.offset = 0
        self.preprocess = preprocess

    def __iter__(self):
        return self

    def __next__(self):
        """Get the next batch (year)

        Returns:
            DataFrame -- Next batch
        """
        if self.offset + self.begin_year == self.end_year:
            raise StopIteration

        # Move to the next year.
        self.offset += 1

        year = str(self.offset + self.begin_year)
        months = [str(x).zfill(2) for x in range(1,13)]
        dataframes = []

        for currency_pair in self.currency_pairs:
            current_dfs = []
            for month in months:
                file_name = f"{currency_pair}-{year}-{month}_15T.csv"
                file_path = path.join(self.location, file_name)
                df = pd.read_csv(file_path)

                # Remove columns.
                del df['open']
                del df['high']
                del df['low']

                # Add this to the current currency pair data frame vertically.
                current_dfs.append(df)

            # Add the dataframe for this currency pair to the list.
            cdf = pd.concat(current_dfs, axis = 0)
            # Remove duplicated ticks.
            cdf.drop_duplicates(subset='tick', inplace=True)

            # The tick should be a datetime.
            cdf['tick'] = pd.to_datetime(cdf['tick'])

            cdf.set_index('tick', inplace=True)
            dataframes.append(cdf)

        # Combine all currency pairs to a single dataframe.
        batch = pd.concat(dataframes, axis = 1)

        # Remove all rows with a null value.
        batch.dropna(inplace=True)

        # Return the current batch if no preprocessing is needed.
        if self.preprocess == False:
            return batch

        # Compute log returns and differences.
        batch = batch.apply(np.log)
        batch = batch.diff()
        batch.dropna(inplace=True)

        # Z-score normalization along each column, with given period
        means = batch.rolling(window=self.period).mean()
        stds = batch.rolling(window=self.period).std()

        batch -= means
        batch /= stds

        # Remove invalid rows.
        batch.dropna(inplace=True)

        return batch

