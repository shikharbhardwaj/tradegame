import pandas as pd
import math
import datetime

data = pd.read_csv("/home/tushar/Desktop/Python/Python3/sampled_data/AUDJPY-2012-01_15T.csv")
for i in range(2,13):
    if i<10:
        m = "0" + str(i)
    else:
        m = str(i)
    df = pd.read_csv(("/home/tushar/Desktop/Python/Python3/sampled_data/AUDJPY-2012-") + m +("_15T.csv"))
    data = data.append(df)

data = data.reset_index()
data = data.drop(['index'],axis = 1)

print("Data Extracted")

def ex_min(row):
    return math.sin((float(row[14:16])/60)*2*3.14)

data["min"] = data["tick"].apply(ex_min)

def ex_hr(row):
    return math.sin((float(row[11:13])/24)*2*3.14)

data["hr"] = data["tick"].apply(ex_hr)

def ex_day(row):
    l = datetime.datetime(int(row[0:4]),int(row[5:7]),int(row[8:10]))
    return math.sin((float(l.weekday())/7)*2*3.14)

data["day_of_week"] = data["tick"].apply(ex_day)

def ex_month(row):
    return math.sin((float(row[5:7])/12)*2*3.14)

data["month"] = data["tick"].apply(ex_month)

print("Time features added")

window_size = 8
datan = data.copy(deep=True)

def exclose(col,window_size,ind):
    m = pd.Series([0]*(window_size-1))
    for i in range(window_size-1,col.shape[0]):
        a = float(col[i-window_size+ind])
        b = float(col[i-window_size+ind+1])
        m = m.append(pd.Series(math.log(b/a)))
    m = m.reset_index()
    return m[0]

datan["close_1"] = exclose(datan["close"],8,1)
datan["close_2"] = exclose(datan["close"],8,2)
datan["close_3"] = exclose(datan["close"],8,3)
datan["close_4"] = exclose(datan["close"],8,4)
datan["close_5"] = exclose(datan["close"],8,5)
datan["close_6"] = exclose(datan["close"],8,6)
datan["close_7"] = exclose(datan["close"],8,7)

print("close features created")

def vol_log(col,window_size,ind):
    m = pd.Series([0]*(window_size-1))
    for i in range(window_size-1,col.shape[0]):
        a = float(col[i-window_size+ind])
        b = float(col[i-window_size+ind+1])
        m = m.append(pd.Series(math.log(b/a)))
    m = m.reset_index()
    return m[0]

datan["volume_1"] = vol_log(datan["volume"],8,1)
datan["volume_2"] = vol_log(datan["volume"],8,2)
datan["volume_3"] = vol_log(datan["volume"],8,3)
datan["volume_4"] = vol_log(datan["volume"],8,4)
datan["volume_5"] = vol_log(datan["volume"],8,5)
datan["volume_6"] = vol_log(datan["volume"],8,6)
datan["volume_7"] = vol_log(datan["volume"],8,7)

print("Volume features created")

data_fin = datan.copy(deep=True)
data_fin = data_fin.drop(['open','high','low','volume'],axis=1)

l = []
for i in range(0,window_size-1):
    l.append(i)

data_fin = data_fin.drop(l)
data_fin.to_csv('/home/tushar/Desktop/Python/Python3/ipython_notebooks/AUSJPY2012_re1.csv',sep='\t',index=False)

print("DONE!!")
