from datetime import datetime, timedelta
from PIL.Image import merge
from pandas.core.reshape.merge import merge_ordered
import customutils as cutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# FUNDAMENTAL DATA
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

dfAdd_betam = "/home/hsl612/gitWorkspace/priceinfo-rl/data/fundamental/beta_market.csv"
dfAdd_compustat = "/home/hsl612/gitWorkspace/priceinfo-rl/data/fundamental/compustatFundVars.csv"
dfAdd_crsp_treasury = "/home/hsl612/gitWorkspace/priceinfo-rl/data/fundamental/treasury.csv"
dfAdd_crsp_marketret = "/home/hsl612/gitWorkspace/priceinfo-rl/data/fundamental/marketret.csv"
dfAdd_wrds_finratio = "/home/hsl612/gitWorkspace/priceinfo-rl/data/fundamental/fundamentals.csv"
dfAdd_marektret = "/home/hsl612/gitWorkspace/priceinfo-rl/data/fundamental/marketret.csv"

df_betam = cutil.cleanDf_betam(dfAdd_betam)
df_compustat = cutil.cleanDf_compustat(dfAdd_compustat)
df_crsptreasury = cutil.cleanDf_crsp_treasury(dfAdd_crsp_treasury)
df_finratio = cutil.cleanDf_wrds_finratio(dfAdd_wrds_finratio)
df_marketret = cutil.cleanDf_crsp_vwret(dfAdd_marektret)

df_1 = cutil.mergeLowerFreqDf(df_high=df_betam, df_low=df_compustat, leftDfName="betasuite", rightDfName="compustat")
df_2 = cutil.mergeLowerFreqDf(df_high=df_1, df_low=df_crsptreasury, leftDfName="df_1", rightDfName="crsp_treasury")
df_3 = cutil.mergeLowerFreqDf(df_high=df_2, df_low=df_finratio, leftDfName="df_2", rightDfName="df_finratio")
df_fdmt = cutil.mergeLowerFreqDf(df_high=df_3, df_low=df_marketret, leftDfName="df_3", rightDfName="df_marketret")

df_fdmt.describe()

df_fdmt["vwretd_annualized"] = df_fdmt["vwretd"]
df_fdmt["t90ret_annualized"] = df_fdmt["t90ret"]
for i_tic in df_fdmt.tic.unique():
    df_fdmt.loc[df_fdmt["tic"] == i_tic, ["vwretd_annualized", "t90ret_annualized"]] = df_fdmt.loc[df_fdmt["tic"] == i_tic, ["vwretd_annualized", "t90ret_annualized"]].rolling(window=252).mean()
df_fdmt["vwretd_annualized"] = (1 + df_fdmt["vwretd_annualized"])**252 - 1# initially at daily
df_fdmt["t90ret_annualized"] = (1 + df_fdmt["t90ret_annualized"])**12 - 1# initially at monthly

# Cost of capital
df_fdmt["ccap_equity"] = df_fdmt["t90ret_annualized"] + df_fdmt["b_mkt"] * (df_fdmt["vwretd_annualized"] - df_fdmt["t90ret_annualized"])
df_fdmt["ccap_debt"] = df_fdmt["xint"] / df_fdmt["dlc"]

# Tobin's Q
df_fdmt["qqq"] = (df_fdmt["at"] + (df_fdmt["csho"]*df_fdmt["prcc_f"]) + df_fdmt["ceq"]) / df_fdmt["at"]

# Short-term assets to total assets
df_fdmt["ccc"] = df_fdmt["che"] / df_fdmt["at"]

df_corefdmt = df_fdmt[["tic", "datadate", "invt_act", "npm", "qqq", "ccc", "ivol", "ccap_equity", "ccap_debt"]]
df_corefdmt.loc[df_corefdmt["ccap_debt"] == np.inf, "ccap_debt"] = np.NaN
for i_tic in df_corefdmt.tic.unique():
    df_corefdmt.loc[df_corefdmt["tic"] == i_tic, ["ccap_debt"]] = df_corefdmt.loc[df_corefdmt["tic"] == i_tic, ["ccap_debt"]].interpolate()
df_corefdmt = df_corefdmt.dropna()

df_corefdmt.describe()

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# PRICE DATA
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

dataAddress = "/home/hsl612/gitWorkspace/priceinfo-rl/data/testData.csv"
unsplitdat = cutil.unsplitPrice(dataAddress)
retdat = cutil.makeRetData(unsplitdat)
retdat.describe()

df_all = pd.merge(left=df_corefdmt, right=retdat, 
    left_on = [df_corefdmt["datadate"].dt.year, df_corefdmt["datadate"].dt.month, df_corefdmt["datadate"].dt.day, df_corefdmt["tic"]],
    right_on = [retdat["datadate"].dt.year, retdat["datadate"].dt.month, retdat["datadate"].dt.day, retdat["tic"]],
    how = "left", validate = "1:1")
df_all = df_all.drop(columns=["key_0", "key_1", "key_2", "key_3", "datadate_y", "tic_y"])
df_all = df_all.rename(columns={"tic_x":"tic", "datadate_x":"datadate"})

df_all.describe()

dfAdd_all = "/home/hsl612/gitWorkspace/priceinfo-rl/data/df_all.csv"
df_all.to_csv(dfAdd_all, index=False)

# ..........................................................................
# VALIDATION

# Plot to check
plt.plot(df_corefdmt.loc[df_corefdmt["tic"] == "AAPL", "ccap_debt"])
plt.show()
plt.plot(df_corefdmt.loc[df_corefdmt["tic"] == "MSFT", "ccap_debt"])
plt.show()

# No missing dates in the middle?
# plt.plot(df_corefdmt.loc[df_corefdmt["tic"] == "MSFT", "datadate"])
# plt.show()

