import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.finance import candlestick_ohlc

df = pd.read_csv('../data/EURUSD-2017-06_24H.csv')

df['tick'] = pd.to_datetime(df['tick'])
df['tick'] = df['tick'].apply(mdates.date2num)

fig = plt.figure()
ax1 = plt.subplot2grid((1, 1), (0,0))
candlestick_ohlc(ax1, df.values, colorup='#53c156', colordown='#ff1717')

ax1.xaxis_date()
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
ax1.grid(True)
    
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('EURUSD-2017-06')
plt.legend()
plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)
plt.show()
