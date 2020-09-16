# Databricks notebook source
from download import download

path = download('http://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv', '/dbfs/mnt/ved-demo/timeseries/appliances/', replace = True)

# COMMAND ----------

import pandas as pd

df = pd.read_csv('/dbfs/mnt/ved-demo/timeseries/appliances/energydata_complete.csv', parse_dates = ['date'])
df = df.set_index('date')
display(df)

# COMMAND ----------

# Impute missing values using forward fill

df = df['Appliances']
df = df.fillna(method="ffill")
df.isnull().sum()

# COMMAND ----------

display(df.reset_index())

# COMMAND ----------

import statsmodels.api as sm

seas_d=sm.tsa.seasonal_decompose(df['2016-01'], model='multiplicative', freq = 24);
fig=seas_d.plot()
fig.set_figheight(15)
fig.set_figwidth(50)
plt.show()

# COMMAND ----------

from statsmodels.tsa.stattools import kpss
import numpy as np

# Null hypothesis: Data is stationary
print(" > Is the data stationary ?")
dftest = kpss(df, 'c')
print("Test statistic = {:.3f}".format(dftest[0]))
print("P-value = {:.3f}".format(dftest[1]))
print("Critical values :")
for k, v in dftest[3].items():
    print("\t{}: {}".format(k, v))

# COMMAND ----------

# MAGIC %md Data is trend stationary

# COMMAND ----------

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df[:100])

# COMMAND ----------

plot_pacf(df[:10])

# COMMAND ----------

