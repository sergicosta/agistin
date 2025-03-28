import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# data: pd.DataFrame() with the form {'x1':{x1_sensitivity_vals...}, 'x2':{x2vals...}, ... 'y':{yvals}}
# y: y column name as str
# base_idx: index of the row that contains the base case as int (default: 0)
def spiderplot(data, y, base_idx=0):
    data_pu = data/data.iloc[base_idx]
    columns = data.drop(y,axis=1).columns
    for c in columns:
        index_plot = (data_pu.drop([c,y],axis=1)==1).all(axis=1).where(lambda x: x).dropna().index
        data_plot = data_pu.iloc[index_plot]
        plt.plot(data_plot[c],data_plot[y],label=c) 
