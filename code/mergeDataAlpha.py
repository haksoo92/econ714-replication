from datetime import datetime, timedelta
from PIL.Image import merge
from pandas.core.reshape.merge import merge_ordered
import customutils as cutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

env_name = "rlagg" # @param {type:"string"}

# OUTPUT DIRECTORY 
env_dir = './out/' + env_name
policy_dir = './out/' + env_name + '/policy' 
returns_dir = './out/' + env_name + '/returns/returns.npy'
rlfolder_dir = './out/' + env_name
plotfolder_dir = './out/' + env_name + '/plot' 

infoLs = ["rlpost_f", "rlpost_p"]
infoLsDisp = ["Fundamental Signal", "Price Signal"]

dfAdd_all = env_dir + "/df_rl.csv"
df_all = pd.read_csv(dfAdd_all)
df_all["datadate"] = pd.to_datetime(df_all["datadate"])

# .......................................................................
# HELPERS
def mergeAlphaDfs(df_high, df_low):
    mergedDf = pd.merge(df_high, df_low, 
                    left_on = [df_high["tic"], df_high["datadate"].dt.year, df_high["datadate"].dt.month, df_high["datadate"].dt.day],
                    right_on = [df_low["tic"], df_low["datadate"].dt.year, df_low["datadate"].dt.month, df_low["datadate"].dt.day],
                    how = "left", validate = "1:1")
    mergedDf = mergedDf.dropna()
    mergedDf = mergedDf.drop(columns=["key_0", "key_1", "key_2", "key_3", "datadate_y", "tic_y"])
    mergedDf = mergedDf.rename(columns={"datadate_x":"datadate", "tic_x":"tic"})
    mergedDf = mergedDf.sort_values(by=["tic", "datadate"])
    return mergedDf

# .......................................................................
dfAdd_alpha_capm = "./out/rlagg/data/capm.csv"
dfAdd_alpha_ff3 = "./out/rlagg/data/ff3.csv"
dfAdd_alpha_ff4 = "./out/rlagg/data/ff4.csv"
dfAdd_alpha_sw = "./out/rlagg/data/sw.csv"

df_alpha_capm = cutil.cleanDf_alpha(dfAdd_alpha_capm, "capm")
df_alpha_ff3 = cutil.cleanDf_alpha(dfAdd_alpha_ff3, "ff3")
df_alpha_ff4 = cutil.cleanDf_alpha(dfAdd_alpha_ff4, "ff4")
df_alpha_sw = cutil.cleanDf_alpha(dfAdd_alpha_sw, "sw")

df_high = mergeAlphaDfs(df_alpha_capm, df_alpha_ff3)
df_high = mergeAlphaDfs(df_high, df_alpha_ff4)
df_high = mergeAlphaDfs(df_high, df_alpha_sw)
df_all_alpha = mergeAlphaDfs(df_high, df_all)

df_all_alpha.to_csv("./out/rlagg/df_rl_alpha.csv", index=False)
