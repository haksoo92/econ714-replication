from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from pandas.core.reshape.merge import merge
import tensorflow as tf

from datetime import datetime
import PIL.Image
import os
import shutil
import tempfile
import zipfile
import base64
import csv

import matplotlib.pyplot as plt

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.policies import policy_saver

from customenv import rlagg
import customutils as cutil
from scipy import stats

tf.compat.v1.enable_v2_behavior()

# TEST SIMPLE LEARNING MODEL (LEARNING FROM PRIVATE SIGNAL)
# .............................................................................
# PARAMETERS
env_name = "rlagg" # @param {type:"string"}

# OUTPUT DIRECTORY 
env_dir = './out/' + env_name
policy_dir = './out/' + env_name + '/policy' 
returns_dir = './out/' + env_name + '/returns/returns.npy'
rlfolder_dir = './out/' + env_name
plotfolder_dir = './out/' + env_name + '/plot' 

# .............................................................................
# EPISODE ANALYSIS

infoLs = ["rlpost_f", "rlpost_p"]
infoLsDisp = ["Fundamental Signal", "Price Signal"]

dfAdd_all = env_dir + "/df_rl.csv"
# df_all = pd.read_csv(dfAdd_all)
df_all = pd.read_csv("./out/rlagg/df_rl_alpha.csv")

# reload the model policy
policy = tf.compat.v2.saved_model.load(policy_dir)

# TASK    train the model
isinit = True
elastic = pd.DataFrame()

# correlation matrix
n_tic, = df_all["tic"].unique().shape
mat_corr = np.zeros((2,n_tic))
mat_elas = np.zeros((2,n_tic))
mat_linreg = np.zeros((4,n_tic))
mat_alphaport = np.zeros((10,n_tic))
ls_tic = []
niter = 0
for currtic in df_all["tic"].unique():
    currdf = df_all[df_all["tic"] == currtic]
    currdate = currdf["datadate"]

    epi_fig = plotfolder_dir + '/episode_' + currtic + '.png'

    eval_py_env = rlagg.PriceInfoEnv(df_all, infoLs, currtic)
    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    eplen = eval_py_env._eplen
    steps = range(eplen)
    oneline = np.ones([eplen+1,1])
    n_factors = len(infoLs)
    
    # ....................................................................
    # DATA GENERATION
    (rlpost, factors, wealth, stockret, tic_date) = cutil.runEpisodesEmpirical2(policy, eval_tf_env, eval_py_env)

    # ...................................................................
    #############################
    ### Compute alpha performance
    #############################
    currdat_alpha = df_all[df_all["tic"] == currtic]
    currdate = pd.DataFrame(tic_date, columns=["tic", "datadate"])
    currdate = currdate.iloc[:,1]
    currdat_alpha["datadate"] = pd.to_datetime(currdat_alpha["datadate"])
    currdf_tic = pd.to_datetime(currdate)

    df_high = currdat_alpha
    df_low = currdf_tic
    mergedDf = pd.merge(df_high, df_low, 
                left_on = [df_high["datadate"].dt.year, df_high["datadate"].dt.month, df_high["datadate"].dt.day],
                right_on = [df_low.dt.year, df_low.dt.month, df_low.dt.day],
                how = "left", validate = "1:1")
    mergedDf = mergedDf.dropna()
    # mergedDf = mergedDf.drop(columns=["key_0", "key_1", "key_2", "key_3", "datadate_y", "tic_y"])
    mergedDf = mergedDf.rename(columns={"datadate_x":"datadate", "tic_x":"tic"})
    mergedDf = mergedDf.sort_values(by=["datadate"])
    alphaDf = mergedDf[["alpha_capm", "alpha_ff3", "alpha_ff4", "alpha_sw"]]

    (rlpost, factors, wealth, stockret, tic_date, meanWealthRet, stdWealthRet, sharpe, skew, kurto, mdd, sharpe_ts, turnover, alpha_capm_ts, alpha_ff3_ts, alpha_ff4_ts, alpha_capm, alpha_ff3, alpha_ff4)  = cutil.getAlphaPortStats(rlpost, factors, wealth, stockret, tic_date, alphaDf)

    currdate = tic_date[:,1]
    currdate = pd.to_datetime(currdate)
    

    # ..................................................................    
    # COMPUTE TABLES

    # Elasticity
    currdat = pd.DataFrame()
    rlpost_percc = rlpost[1:,0]/rlpost[0:-1,0] - 1
    rlpostf_percc = factors[1:,0]/factors[0:-1,0] - 1
    rlpostp_percc = factors[1:,1]/factors[0:-1,1] - 1
    currdat = pd.DataFrame(np.column_stack([rlpost_percc, rlpostf_percc, rlpostp_percc]), columns=["rlpost", "rlpost_f", "rlpost_p"])
    if isinit:
        elastic = currdat
        isinit = False
    else:
        elastic = pd.concat([elastic, currdat], axis=0, ignore_index=True)
    elas_f =  elastic["rlpost"]/elastic["rlpost_f"]
    elas_p =  elastic["rlpost"]/elastic["rlpost_p"]
    elas_f = np.round(np.mean(elas_f), decimals=2)
    elas_p = np.round(np.mean(elas_p), decimals=2)
    mat_elas[0,niter] = elas_f
    mat_elas[1,niter] = elas_p

    # ...................................................................
    # PLOT FIGURES
    currdate = currdate.date
    nd = 300
    fig, axs = plt.subplots(4)
    steps = steps[0:np.size(stockret)]
    axs[0].title.set_text(currtic)
    axs[0].plot(steps, stockret, 'k--', label='Return Realized (Daily)', linewidth=1.0)
    axs[0].set_xticks(steps[::nd])
    axs[0].set_xticklabels(currdate[::nd])
    axs[0].set_ylabel("Realized Returns (Daily)")
    axtwin = axs[0].twinx()
    axtwin.plot([-1] + [*steps], wealth, 'r-', label='Wealth')
    axtwin.set_ylabel("Wealth")
    axtwin.plot([-1] + [*steps], np.ones(np.shape(wealth)), 'k-', linewidth=1.5)
    axs[0].legend(loc='lower right')
    axtwin.legend(loc='lower left')

    # PANEL: Portfolio weight
    axs[1].plot(steps, rlpost, 'k-', label='RL Portfolio Weight')
    axs[1].set_xticks(steps[::nd])
    axs[1].set_xticklabels(currdate[::nd])
    plt.setp(axs[1], ylabel='RL Portfolio Weight')
    axs[1].legend(loc='lower right')

    # PANEL: Sharpe
    axs[2].plot(steps, sharpe_ts, 'k-', label='RL Sharpe')
    axs[2].set_xticks(steps[::nd])
    axs[2].set_xticklabels(currdate[::nd])
    plt.setp(axs[2], ylabel='Sharpe')
    axs[2].legend(loc='lower right')

    # PANEL: Alpha
    axs[3].plot(steps, alpha_capm_ts*100, 'k-', label='CAPM')
    axs[3].plot(steps, alpha_ff3_ts*100, 'k--', label='FF3')
    axs[3].plot(steps, alpha_ff4_ts*100, 'k:', label='FF4')
    axs[3].set_xticks(steps[::nd])
    axs[3].set_xticklabels(currdate[::nd])
    plt.setp(axs[3], ylabel='Alpha (%)')
    axs[3].legend(loc='lower right')

    # for i in range(n_factors):
    #     axs[2+i].plot(steps, factors[:,i], 'g--', label=infoLsDisp[i])
    #     axs[2+i].set_xticks(steps[::nd])
    #     axs[2+i].set_xticklabels(currdate[::nd])
    #     plt.setp(axs[2+i], ylabel=infoLsDisp[i])
    #     axs[2+i].legend(loc='lower right')

    fig.set_size_inches(12, 6 + 3*n_factors + 2)
    plt.ylim()
    # plt.show()
    # plt.close()
    plt.savefig(epi_fig)
    plt.close()

    # .....................................................................
    # COMPUTE CORRELATIONS
    corr_f = np.corrcoef(rlpost[:,0], factors[:,0])
    corr_f = np.round(corr_f[0,1], decimals=2) 
    corr_p = np.corrcoef(rlpost[:,0], factors[:,1])
    corr_p = np.round(corr_p[0,1], decimals=2)
    # note that the RL agent bases her decision on 63-trading days signals
    ls_tic.append(currtic)
    mat_corr[0,niter] = corr_f
    mat_corr[1,niter] = corr_p

    # .....................................................................
    # COMPUTE BETAS
    linreg_res = stats.linregress(factors[:,0], rlpost[:,0])
    mat_linreg[0,niter] = linreg_res.slope
    mat_linreg[1,niter] = linreg_res.stderr
    mat_linreg[2,niter] = linreg_res.intercept
    mat_linreg[3,niter] = linreg_res.intercept_stderr
    mat_linreg = np.round(mat_linreg, decimals=4)

    # .....................................................................
    # COMPUTE ALPHA PORTFOLIO STATS
    mat_alphaport[0,niter] = meanWealthRet
    mat_alphaport[1,niter] = stdWealthRet
    mat_alphaport[2,niter] = sharpe
    mat_alphaport[3,niter] = skew
    mat_alphaport[4,niter] = kurto
    mat_alphaport[5,niter] = turnover
    mat_alphaport[6,niter] = mdd
    mat_alphaport[7,niter] = alpha_capm * 100
    mat_alphaport[8,niter] = alpha_ff3 * 100
    mat_alphaport[9,niter] = alpha_ff4 * 100
    mat_alphaport = np.round(mat_alphaport, decimals=2)
    niter += 1

# format correlation mat
df_corr = pd.DataFrame(mat_corr, ["corr_f", "corr_p"], ls_tic)
dir_corrmat = plotfolder_dir + '/corrmat.csv'
df_corr.to_csv(dir_corrmat)

# format elasticity mat
df_elas = pd.DataFrame(mat_elas, ["elasticity_f", "elasticity_p"], ls_tic)
dir_elasmat = plotfolder_dir + '/elasmat.csv'
df_elas.to_csv(dir_elasmat)

# format price inform mat
df_pi = pd.DataFrame(mat_linreg, ["beta", "beta_stderr", "alpha", "alpha_stderr"], ls_tic)
dir_pi = plotfolder_dir + '/priceinfomat.csv'
df_pi.to_csv(dir_pi)

# format alpha portfolio mat
df_alphaport = pd.DataFrame(mat_alphaport, ["Return", "Std", "Sharpe", "Skewness", "Kurtosis", "Turnover", "MDD", "Alpha CAPM", "Alpha FF3", "Alpha FF4"], ls_tic)
dir_ap = plotfolder_dir + '/alphaPortfolio.csv'
df_alphaport.to_csv(dir_ap)

# plt.plot(currdat.iloc[0:100,0], label="rlpost")
# plt.plot(currdat.iloc[0:100,1], label="funda")
# plt.plot(currdat.iloc[0:100,2], label="price")
# plt.legend()
# plt.show()
# ax = plt.axes(projection='3d')
# ax.scatter3D(elastic["rlpost_f"]*100, elastic["rlpost_p"]*100, elastic["rlpost"]/elastic["rlpost_f"], c = elastic["rlpost"]/elastic["rlpost_f"], cmap='Greens')
# plt.show()

# .............................................................................
# TEST FOR CONVERGENCE
import json
with open(policy_dir + "/hyperparams.json") as f:
    hparam = json.load(f)
eval_interval = hparam['eval_interval']

returns_fig = plotfolder_dir + '/returns.png'
with open(returns_dir, 'rb') as f:
    returns = np.load(f)

plt.clf()
fig, axs = plt.subplots(1)
steps = (np.array([*range(0, len(returns))]) + 1) * eval_interval
axs.plot(steps, returns, 'k')
plt.setp(axs, xlabel='Iteration')
plt.setp(axs, ylabel='Average Return')

fig.set_size_inches(12, 5)
plt.ylim()
# plt.show()
plt.savefig(returns_fig)
plt.close()