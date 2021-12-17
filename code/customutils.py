from datetime import datetime, timedelta
from mmap import ACCESS_DEFAULT
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import scipy

def cleanDf_wrds_finratio(datfdmt_add):
    dat = pd.read_csv(datfdmt_add)
    dat["public_date"] = pd.to_datetime(dat["public_date"])
    varli_fdmt = ["bm", "npm", "capital_ratio", "invt_act", "de_ratio", "inv_turn"]
    dat = dat[varli_fdmt + ["TICKER", "public_date"]]
    dat = dat.rename(columns={"TICKER":"tic", "public_date":"datadate", "capital_ratio":"cap_ratio"})
    dat = dat.sort_values(by=["tic", "datadate"])
    dat["datadate"] = dat["datadate"] + timedelta(days=1) # data not available during the uneditted month
    return dat


def mergePriceFdmt(dat_ret, dat_fdmt):
    mergedDf = pd.merge(dat_fdmt, dat_ret, 
        left_on = [dat_fdmt["tic"], dat_fdmt["datadate"].dt.year, dat_fdmt["datadate"].dt.month],
        right_on = [dat_ret["tic"], dat_ret["datadate"].dt.year, dat_ret["datadate"].dt.month],
        how = "right")
    keepli = ['bm', 'npm', 'cap_ratio', 'invt_act','de_ratio', 'inv_turn', 
                'prccd', 'prchd','prcld', 'prcod', 'datadate_y', 'tic_y']
    mergedDf = mergedDf[keepli]
    mergedDf = mergedDf.rename(columns={"datadate_y":"datadate", "tic_y":"tic"})
    mergedDf = mergedDf.dropna()
    mergedDf = mergedDf.sort_values(by = ["tic","datadate"])
    return mergedDf


def unsplitPrice(dataAddress):
    dat = pd.read_csv(dataAddress)
    dat.datadate = pd.to_datetime(dat.datadate)
    editli = ["prccd", "prchd", "prcld", "prcod"]

    # AAPLplt.show()

    dat.loc[(dat["datadate"] >= '2014-06-9') & (dat["tic"] == "AAPL"), editli] *= 7
    dat.loc[(dat["datadate"] >= '2020-08-29') & (dat["tic"] == "AAPL"), editli] *= 4

    # MSFT

    # TSLA
    dat.loc[(dat["datadate"] >= '2020-08-31') & (dat["tic"] == "TSLA"), editli] *= 5

    return dat


def makeRetData(dat):
    """
    Make into return data

    :param dat : pandas
    :return : pandas
    """
    dat.datadate = pd.to_datetime(dat.datadate)
    dat.sort_values(by=["tic", "datadate"])

    keptli = ["datadate", "tic", "prccd", "prchd", "prcld", "prcod"]
    editli = ["prccd", "prchd", "prcld", "prcod"]
    dat = dat[keptli]

    isinit = True
    for i_tic in dat.tic.unique():
        currdat = dat[dat.tic == i_tic]
        newdat = currdat.copy()
        newdat = newdat[editli]
        olddat = newdat.copy()
        n_i, n_j = olddat.shape
        for i in range(n_i):
            if i == 0:
                continue
            else:
                for j in range(n_j):
                    newdat.iloc[i,j] = float(olddat.iloc[i,j]) / float(olddat.iloc[i-1,j])
        newdat["datadate"] = dat["datadate"]
        newdat["tic"] = dat["tic"]
        newdat = newdat.iloc[1:,:]
        if isinit:
            resdat = newdat
            isinit = False
        else:
            resdat = resdat.append(newdat)
    return resdat


def getEpisodePriceData(dat):
    """
    Get a relevant data period. Select a random traiding date uniformly (t),
    and get X for t+252 and X_0 for t-63

    :param dat : pd, data

    :return (x_m3mo, x_p12mo) : (numpy ndarray) initial input, following 12-mo outcomes
    """
    ticli = dat.tic.unique()
    ntic, = ticli.shape
    rnd_ticidx = np.random.randint(0, ntic-1)
    currtic = ticli[rnd_ticidx]
    dat = dat[dat.tic == currtic]

    eplen = 252
    inputlen = 63
    featurelen = 4

    uniqTradingDays = dat.datadate.unique()
    n_td, = uniqTradingDays.shape
    lb = inputlen + 1
    ub = n_td - 1 - eplen
    rnd_tdidx = np.random.randint(lb, ub)

    priceInfoLs = ["prccd", "prchd", "prcld", "prcod"]
    dat = dat[priceInfoLs]
    currdat = dat.iloc[rnd_tdidx-inputlen+1:rnd_tdidx+eplen+1, :]

    x_m3mo = currdat.iloc[0:inputlen]
    x_p12mo = currdat.iloc[inputlen:]

    if (not x_m3mo.shape == (inputlen, featurelen)) or (not x_p12mo.shape == (eplen, featurelen)):
        raise ValueError

    return (x_m3mo.to_numpy(), x_p12mo.to_numpy())


def getEpisodeData(dat, infoLs):
    """
    Get a relevant data period. Select a random traiding date uniformly (t),
    and get X for t+252 and X_0 for t-63

    :param dat : pd, data

    :return (x_m3mo, x_p12mo) : (numpy ndarray) initial input, following 12-mo outcomes
    """
    ticli = dat.tic.unique()
    ntic, = ticli.shape
    rnd_ticidx = np.random.randint(0, ntic-1)
    currtic = ticli[rnd_ticidx]
    dat = dat[dat.tic == currtic]

    eplen = 252
    inputlen = 63
    featurelen = len(infoLs)

    uniqTradingDays = dat.datadate.unique()
    n_td, = uniqTradingDays.shape
    lb = inputlen + 1
    ub = n_td - 1 - eplen
    rnd_tdidx = np.random.randint(lb, ub)

    dat = dat[infoLs]
    currdat = dat.iloc[rnd_tdidx-inputlen+1:rnd_tdidx+eplen+1, :]

    x_m3mo = currdat.iloc[0:inputlen]
    x_p12mo = currdat.iloc[inputlen:]

    if (not x_m3mo.shape == (inputlen, featurelen)) or (not x_p12mo.shape == (eplen, featurelen)):
        raise ValueError

    return (x_m3mo, x_p12mo)


def getFullEpisodeData(dat, currtic, infoLs):
    """
    Get a relevant data period, largest possible

    :param dat : pd, data
    :return (x_m3mo, x_p12mo) : (numpy ndarray) initial input, data to the end
    """
    # ticli = dat.tic.unique()
    # ntic, = ticli.shape
    # rnd_ticidx = 0
    # currtic = ticli[rnd_ticidx]
    dat = dat[dat.tic == currtic]

    # eplen = 252
    inputlen = 63
    featurelen = len(infoLs)

    uniqTradingDays = dat.datadate.unique()
    n_td, = uniqTradingDays.shape
    lb = inputlen + 1
    # ub = n_td - 1 - eplen
    rnd_tdidx = lb

    dat = dat[infoLs]
    currdat = dat.iloc[rnd_tdidx-inputlen+1:, :]

    x_m3mo = currdat.iloc[0:inputlen]
    x_p12mo = currdat.iloc[inputlen:]

    # if (not x_m3mo.shape == (inputlen, featurelen)) or (not x_p12mo.shape == (eplen, featurelen)):
    #     raise ValueError

    return (x_m3mo, x_p12mo)

# # Unit tests
# dataAddress = "/home/hsl612/gitWorkspace/priceinfo-rl/data/testData.csv"
# unsplitdat = unsplitPrice(dataAddress)
# dat_ret = makedat_reta(unsplitdat)
# x1, x2 = getEpisodePriceData(dat_ret)

# plt.plot(dat_ret[dat_ret.tic == "AAPL"].prccd)
# plt.plot(unsplitdat[unsplitdat.tic == "AAPL"].prccd)
# plt.show()

# plt.plot(dat_ret[dat_ret.tic == "MSFT"].prccd)
# plt.plot(unsplitdat[unsplitdat.tic == "MSFT"].prccd)
# plt.show()

# plt.plot(dat_ret[dat_ret.tic == "TSLA"].prccd)
# plt.plot(unsplitdat[unsplitdat.tic == "TSLA"].prccd)
# plt.show()


def getCurrWealth(prevWealth, action, currReturn):
    """
    Computes the current wealth from portfolio weights and stock return

    :param prevWealth : float, prev wealth level
    :param action : float, [0,1], weight on the stock
    :param currReturn : float, return on the stock

    :return currWealth : float, current wealth
    """
    currRf = 1.0 # current risk-free returns
    currWealth = (currReturn * (prevWealth * action)) + (currRf * (prevWealth * (1-action)))

    return currWealth


def getPortRet(traj_wealth):
    """
    Computes the returns trajectory
    """
    eplen, = traj_wealth.shape
    traj_ret = np.zeros(eplen)
    traj_ret[0] = 1
    for i in range(eplen - 1):
        traj_ret[i+1] = traj_wealth[i+1] / traj_wealth[i]
    return traj_ret

"""
# Tests
dataAddress = "/home/hsl612/gitWorkspace/priceinfo-rl/data/testData.csv"
x_m3mo, x_p12mo = getEpisodePriceData(dataAddress)
traj_wealth = x_m3mo[:,0]
getPortRet(traj_wealth)
"""


def getSharpe(traj_wealth, rf) :
    """ 
    Computes the Sharpe ratio of a portfolio over a given period

    :param traj_wealth : (eplen + 1, ) double, wealth level of portfolio 
    :param rf : double, risk-free rate (daily)

    :return : double, Sharpe ratio at daily frequency
    """
    traj_ret = getPortRet(traj_wealth)
    traj_extret = traj_ret - rf
    sharpe = np.mean(traj_extret) / np.std(traj_extret)

    return sharpe

"""
# Tests 
rf = 1.0
getSharpe(traj_wealth, rf) # note annualizing the daily Sharpe to annual Sharpe is not trivial 
"""

def getModifiedSharpe(traj_wealth) :
    """ 
    Computes the Sharpe ratio of a wealth over a given period.
    Average-level-of-wealth-per-period / sd-of-wealth-level

    :param traj_wealth : (eplen + 1, ) double, wealth level of portfolio 
    :param rf : double, risk-free rate (daily)

    :return : double, Sharpe ratio at daily frequency
    """
    sharpe = np.mean(traj_wealth) / np.std(traj_wealth)

    return sharpe


def getExpectedReturn(traj_wealth, rf) :
    """ 
    Computes the Sharpe ratio of a portfolio over a given period

    :param traj_wealth : (eplen + 1, ) double, wealth level of portfolio 
    :param rf : double, risk-free rate (daily)

    :return : double, Sharpe ratio at daily frequency
    """
    traj_ret = getPortRet(traj_wealth)
    traj_extret = traj_ret - rf
    sharpe = np.mean(traj_extret) / np.std(traj_extret)

    return sharpe

# CRRA
# http://karlshell.com/wp-content/uploads/2015/09/WebPage.pdf
# CRRA estimates [1, 2, 5, 30], mean at 2, log at 1, max 5



def getBayes(sig_mu, sig_sd, prior_mu, prior_sd):
    sig = sig_mu
    sigPrec = 1/(sig_sd**2)
    prior = prior_mu
    priorPrec = 1/(prior_sd**2)
    
    # Bayes' posterior mean
    bayes_mu = (1/(sigPrec + priorPrec))*(sigPrec*sig + priorPrec*prior)

    # Bayes' posterior variance
    bayes_sig = np.asscalar(np.sqrt((priorPrec + sigPrec)**(-1)))

    return (bayes_mu, bayes_sig)

def getPrice(ev1, ev2, sdv1, supply, gamma, priceImpact_bayes):
    p = (((1-priceImpact_bayes)*ev1)+(priceImpact_bayes*ev2)) - gamma*(sdv1**2)*supply
    return p

def runEpisodes_basic(policy, eval_tf_env, eval_py_env):

    num_episodes = 1
    eplen = 50 # change this
    bayespost = np.empty([eplen,1], dtype=np.float32)
    rlpost = np.empty([eplen,1], dtype=np.float32)
    fundamental = np.empty([eplen,1], dtype=np.float32)
    # surprise = np.empty([eplen,1], dtype=np.float32)

    for _ in range(num_episodes):
        time_step = eval_tf_env.reset()
        f = eval_py_env._f
        for i in range(eplen):

            action_step = policy.action(time_step)
            time_step = eval_tf_env.step(action_step.action)

            curr_obs = time_step.observation.numpy()[0]

            # TASK   compute the Bayes posterior
            sig = curr_obs[0]
            sigPrec = 1/(curr_obs[1]**2)
            prior = curr_obs[2]
            priorPrec = 1/(curr_obs[3]**2)
            curr_bpost = (1/(sigPrec + priorPrec))*(sigPrec*sig + priorPrec*prior)
            bayespost[i] = curr_bpost

            # TASK    get the RL agent action
            curr_action = action_step.action[0].numpy()
            rlpost[i] = curr_action

            # TASK  fill in the funamental
            fundamental[i] = f

            # TASK  fill in surprise
            # surprise[i] = curr_obs[5]
    
    return (rlpost, bayespost, fundamental)

def runEpisodes_basic_dqn(policy, eval_tf_env, action_map):

    num_episodes = 1
    eplen = 50 # change this
    bayespost = np.empty([eplen,1], dtype=np.float32)
    rlpost = np.empty([eplen,1], dtype=np.float32)
    fundamental = np.empty([eplen,1], dtype=np.float32)
    # surprise = np.empty([eplen,1], dtype=np.float32)

    for _ in range(num_episodes):
        time_step = eval_tf_env.reset()
        f = eval_tf_env.envs[0]._f
        for i in range(eplen):

            action_step = policy.action(time_step)
            time_step = eval_tf_env.step(action_step.action)

            curr_obs = time_step.observation.numpy()[0]

            # TASK   compute the Bayes posterior
            sig = curr_obs[0]
            sigPrec = 1/(curr_obs[1]**2)
            prior = curr_obs[2]
            priorPrec = 1/(curr_obs[3]**2)
            curr_bpost = (1/(sigPrec + priorPrec))*(sigPrec*sig + priorPrec*prior)
            bayespost[i] = curr_bpost

            # TASK    get the RL agent action
            curr_action = action_step.action[0].numpy()
            rlpost[i] = action_map[curr_action]

            # TASK  fill in the funamental
            fundamental[i] = f

            # TASK  fill in surprise
            # surprise[i] = curr_obs[5]
    
    return (rlpost, bayespost, fundamental)

def runEpisodes_basic_price_dqn(policy, eval_tf_env, action_map):

    num_episodes = 1
    eplen = 50 # change this
    bayespost = np.empty([eplen,1], dtype=np.float32)
    rlpost = np.empty([eplen,1], dtype=np.float32)
    fundamental = np.empty([eplen,1], dtype=np.float32)
    price = np.empty([eplen,1], dtype=np.float32)

    for _ in range(num_episodes):
        time_step = eval_tf_env.reset()
        f = eval_tf_env.envs[0]._f
        for i in range(eplen):

            action_step = policy.action(time_step)
            time_step = eval_tf_env.step(action_step.action)

            curr_obs = time_step.observation.numpy()[0]

            # TASK   compute the Bayes posterior
            sig = curr_obs[0]
            sigPrec = 1/(curr_obs[1]**2)
            prior = curr_obs[2]
            priorPrec = 1/(curr_obs[3]**2)
            curr_bpost = (1/(sigPrec + priorPrec))*(sigPrec*sig + priorPrec*prior)
            bayespost[i] = curr_bpost

            # TASK    get the RL agent action
            curr_action = action_step.action[0].numpy()
            rlpost[i] = action_map[curr_action]

            # TASK  fill in the funamental
            fundamental[i] = f

            # TASK  fill in price
            price[i] = curr_obs[4]
    
    return (rlpost, bayespost, fundamental, price)

def runEpisodes_basic_gs7(policy, eval_tf_env, eval_py_env):

    num_episodes = 1
    eplen = 50 # change this
    bayespost = np.empty([eplen,1], dtype=np.float32)
    rlpost = np.empty([eplen,1], dtype=np.float32)
    fundamental = np.empty([eplen,1], dtype=np.float32)
    # surprise = np.empty([eplen,1], dtype=np.float32)

    for _ in range(num_episodes):
        time_step = eval_tf_env.reset()
        f = eval_py_env._mu
        for i in range(eplen):

            action_step = policy.action(time_step)
            time_step = eval_tf_env.step(action_step.action)

            curr_obs = time_step.observation.numpy()[0]

            # TASK   compute the Bayes posterior
            sig = curr_obs[0]
            sigPrec = 1/(curr_obs[1]**2)
            prior = curr_obs[2]
            priorPrec = 1/(curr_obs[3]**2)
            curr_bpost = (1/(sigPrec + priorPrec))*(sigPrec*sig + priorPrec*prior)
            bayespost[i] = curr_bpost

            # TASK    get the RL agent action
            curr_action = action_step.action[0].numpy()
            rlpost[i] = curr_action

            # TASK  fill in the funamental
            fundamental[i] = f

            # TASK  fill in surprise
            # surprise[i] = curr_obs[5]
    
    return (rlpost, bayespost, fundamental)

# def runEpisodes(policy, eval_tf_env, eval_py_env):

#     num_episodes = 1
#     eplen = 50 # change this
#     bayespost = np.empty([eplen,1], dtype=np.float32)
#     rlpost = np.empty([eplen,1], dtype=np.float32)
#     fundamental = np.empty([eplen,1], dtype=np.float32)
#     surprise = np.empty([eplen,1], dtype=np.float32)

#     for _ in range(num_episodes):
#         time_step = eval_tf_env.reset()
#         f = eval_py_env._f
#         for i in range(eplen):

#             action_step = policy.action(time_step)
#             time_step = eval_tf_env.step(action_step.action)

#             curr_obs = time_step.observation.numpy()[0]

#             # TASK   compute the Bayes posterior
#             sig = curr_obs[0]
#             sigPrec = 1/(curr_obs[1]**2)
#             prior = curr_obs[2]
#             priorPrec = 1/(curr_obs[3]**2)
#             curr_bpost = (1/(sigPrec + priorPrec))*(sigPrec*sig + priorPrec*prior)
#             bayespost[i] = curr_bpost

#             # TASK    get the RL agent action
#             curr_action = action_step.action[0].numpy()
#             rlpost[i] = curr_action

#             # TASK  fill in the funamental
#             fundamental[i] = f

#             # TASK  fill in surprise
#             surprise[i] = curr_obs[5]
    
#     return (rlpost, bayespost, fundamental, surprise)

def runEpisodes(policy, eval_tf_env, eval_py_env):

    num_episodes = 1
    eplen = 50 # change this
    bayespost = np.empty([eplen,1], dtype=np.float32)
    rlpost = np.empty([eplen,1], dtype=np.float32)
    fundamental = np.empty([eplen,1], dtype=np.float32)
    surprise = np.empty([eplen,1], dtype=np.float32)
    refbelief = np.empty([eplen,1], dtype=np.float32)
    # var6 = np.empty([eplen,1], dtype=np.float32)
    # var7 = np.empty([eplen,1], dtype=np.float32)
    
    for _ in range(num_episodes):
        time_step = eval_tf_env.reset()
        f = eval_py_env._f
        for i in range(eplen):
            action_step = policy.action(time_step)
            time_step = eval_tf_env.step(action_step.action)

            curr_obs = time_step.observation.numpy()[0]

            # TASK   compute the Bayes posterior
            sig = curr_obs[0]
            sigPrec = 1/(curr_obs[1]**2)
            prior = curr_obs[2]
            priorPrec = 1/(curr_obs[3]**2)
            curr_bpost = (1/(sigPrec + priorPrec))*(sigPrec*sig + priorPrec*prior)
            bayespost[i] = curr_bpost

            # TASK    get the RL agent action
            curraction_step = policy.action(time_step)
            curr_action = curraction_step.action[0].numpy()
            rlpost[i] = curr_action

            # TASK  fill in the funamental
            fundamental[i] = f

            # TASK  fill in surprise
            surprise[i] = curr_obs[5]

            refbelief[i] = curr_obs[4]
            # var6[i] = curr_obs[6]
            # var7[i] = curr_obs[7]
    
    return (rlpost, bayespost, fundamental, surprise, refbelief)


def runEpisodesEmpirical(policy, eval_tf_env, eval_py_env):
    num_episodes = 1
    eplen = eval_py_env._eplen
    rlpost = np.empty([eplen,1], dtype=np.float32)
    factors = np.empty([eplen, eval_py_env._n_state], dtype=np.float32)
    for _ in range(num_episodes):
        time_step = eval_tf_env.reset()
        for i in range(eplen):
            action_step = policy.action(time_step)
            time_step = eval_tf_env.step(action_step.action)
            curr_obs = time_step.observation.numpy()[0]
            # RL agent action
            curraction_step = policy.action(time_step)
            curr_action = curraction_step.action[0].numpy()
            rlpost[i] = curr_action
            # Input factors
            factors[i,:] = curr_obs[-1,:]
        wealth = eval_py_env._traj_wealth
        stockret = eval_py_env._epdat_y
    return (rlpost, factors, wealth, stockret)

def runEpisodesEmpirical2(policy, eval_tf_env, eval_py_env):
    num_episodes = 1
    eplen = eval_py_env._eplen
    rlpost = np.empty([eplen,1], dtype=np.float32)
    factors = np.empty([eplen, eval_py_env._n_state], dtype=np.float32)
    for _ in range(num_episodes):
        time_step = eval_tf_env.reset()
        for i in range(eplen):
            action_step = policy.action(time_step)
            time_step = eval_tf_env.step(action_step.action)
            curr_obs = time_step.observation.numpy()[0]
            # RL agent action
            curraction_step = policy.action(time_step)
            curr_action = curraction_step.action[0].numpy()
            rlpost[i] = curr_action
            # Input factors
            factors[i,:] = curr_obs[-1,:]
        wealth = eval_py_env._traj_wealth
        stockret = eval_py_env._epdat_y
        tic_key = eval_py_env._epdat_key
    return (rlpost, factors, wealth, stockret, tic_key)

def getAlphaPortStats(rlpost, factors, wealth, stockret, tic_date, df_alpha):
    # Time series outputs
    df_alpha = df_alpha.reset_index()
    wealthDailyRet = np.zeros(np.shape(wealth))
    alpha_capm = np.zeros(np.shape(wealth))
    alpha_ff3 = np.zeros(np.shape(wealth))
    alpha_ff4 = np.zeros(np.shape(wealth))
    alpha_sw = np.zeros(np.shape(wealth))
    for i in range(np.size(wealthDailyRet) - 1):
        wealthDailyRet[i] = wealth[i+1]/wealth[i]
        alpha_capm[i] = rlpost[i]* df_alpha.loc[i, "alpha_capm"]
        alpha_ff3[i] = rlpost[i]* df_alpha.loc[i, "alpha_ff3"]
        alpha_ff4[i] = rlpost[i]* df_alpha.loc[i, "alpha_ff4"]
        alpha_sw[i] = rlpost[i]* df_alpha.loc[i, "alpha_sw"]
    wealthDailyRet = wealthDailyRet[0:-1]
    alpha_capm = alpha_capm[0:-1]
    alpha_ff3 = alpha_ff3[0:-1]
    alpha_ff4 = alpha_ff4[0:-1]
    alpha_sw = alpha_sw[0:-1]
    
    sharpe_ts =  np.zeros((np.size(wealthDailyRet) - 252,))
    alpha_capm_ts = np.zeros((np.size(wealthDailyRet) - 252,))
    alpha_ff3_ts = np.zeros((np.size(wealthDailyRet) - 252,))
    alpha_ff4_ts = np.zeros((np.size(wealthDailyRet) - 252,))
    alpha_sw_ts = np.zeros((np.size(wealthDailyRet) - 252,))
    for i in range(np.size(wealthDailyRet) - 252):
        curr = wealthDailyRet[0+i:252+i]
        curr_alpha_capm = alpha_capm[0+i:252+i] + 1
        curr_alpha_ff3 = alpha_ff3[0+i:252+i] + 1
        curr_alpha_ff4 = alpha_ff4[0+i:252+i] + 1
        curr_alpha_sw = alpha_sw[0+i:252+i] + 1

        sharpe_ts[i] = ((np.mean(curr)-1)/np.std(curr))*np.sqrt(252)
        alpha_capm_ts[i] = np.prod(curr_alpha_capm) - 1
        alpha_ff3_ts[i] = np.prod(curr_alpha_ff3) - 1
        alpha_ff4_ts[i] = np.prod(curr_alpha_ff4) - 1
        alpha_sw_ts[i] = np.prod(curr_alpha_sw) - 1
    rlpost = rlpost[252:]
    factors = factors[252:,:]
    wealth = wealth[252:,]
    stockret = stockret[252:,]
    tic_date = tic_date[252:,]
    # Alpha means
    alpha_capm = np.mean(alpha_capm_ts)
    alpha_ff3 = np.mean(alpha_ff3_ts)
    alpha_ff4 = np.mean(alpha_ff4_ts)

    # Scalar
    # Annualized returns
    wealthret = np.zeros(np.shape(wealth[252:]))
    for i in range(np.size(wealthret)):
        wealthret[i] = wealth[252+i]/wealth[0]
    meanWealthRet = np.mean(wealthret)
    stdWealthRet = np.std(wealthret)
    sharpe = meanWealthRet/stdWealthRet
    skew = stats.skew(wealthret)
    kurto = stats.kurtosis(wealthret)
    # MDD
    i = 0
    mdd_ts = np.zeros(np.size(wealth[251:]))
    for _ in wealth[251:]:
        currMax = np.max(wealth[i:251+i])
        currMin = np.min(wealth[i:251+i])
        mdd_ts[i] = (currMax-currMin) / currMax
        i += 1
    mdd = np.mean(mdd_ts)
    # Turnover
    turnover = np.zeros((np.size(rlpost)-1,))
    for i in range(np.size(rlpost)-1):
        turnover[i] = np.abs(rlpost[i] - rlpost[i-1]) * 252
    turnover = np.mean(turnover)
    return (rlpost, factors, wealth, stockret, tic_date, meanWealthRet, stdWealthRet, sharpe, skew, kurto, mdd, sharpe_ts, turnover, alpha_capm_ts, alpha_ff3_ts, alpha_ff4_ts, alpha_capm, alpha_ff3, alpha_ff4) 

def runEpisodesTheo(policy, eval_tf_env, eval_py_env):
    num_episodes = 1
    eplen = eval_py_env._eplen
    rlpost = np.empty([eplen,1], dtype=np.float32)
    factors = np.empty([eplen, eval_py_env._n_state], dtype=np.float32)
    for _ in range(num_episodes):
        time_step = eval_tf_env.reset()
        for i in range(eplen):
            action_step = policy.action(time_step)
            time_step = eval_tf_env.step(action_step.action)
            curr_obs = time_step.observation.numpy()[0]
            # RL agent action
            curraction_step = policy.action(time_step)
            curr_action = curraction_step.action[0].numpy()
            rlpost[i] = curr_action
            # Input factors
            factors[i,:] = curr_obs[-1,:]
        wealth = eval_py_env._traj_wealth
        belief = eval_py_env._belief
        belief_oppo = eval_py_env._belief_oppo
        demand = eval_py_env._demand
        filteredSignal = eval_py_env._filteredSignal
        filteredSignal_oppo = eval_py_env._filteredSignal_oppo
        # stockret = eval_py_env._epdat_y
    return (rlpost, factors, wealth, belief, belief_oppo, demand, filteredSignal, filteredSignal_oppo)


def runEpisodesTheo2(policy, eval_tf_env, eval_py_env):
    num_episodes = 1
    eplen = eval_py_env._eplen
    rlpost = np.empty([eplen,1], dtype=np.float32)
    factors = np.empty([eplen, eval_py_env._n_state], dtype=np.float32)
    for _ in range(num_episodes):
        time_step = eval_tf_env.reset()
        for i in range(eplen):
            action_step = policy.action(time_step)
            time_step = eval_tf_env.step(action_step.action)
            curr_obs = time_step.observation.numpy()[0]
            curr_obs = np.reshape(curr_obs, eval_py_env._state.shape)
            # RL agent action
            curraction_step = policy.action(time_step)
            curr_action = curraction_step.action[0].numpy()
            rlpost[i] = curr_action
            # Input factors
            factors[i,:] = curr_obs[-1,:]
        wealth = eval_py_env._traj_wealth
        belief = eval_py_env._belief
        belief_oppo = eval_py_env._belief_oppo
        demand = eval_py_env._demand
        filteredSignal = eval_py_env._filteredSignal
        filteredSignal_oppo = eval_py_env._filteredSignal_oppo
        # stockret = eval_py_env._epdat_y
    return (rlpost, factors, wealth, belief, belief_oppo, demand, filteredSignal, filteredSignal_oppo)

def runFullEpisodesEmpirical(policy, eval_tf_env, eval_py_env):
    num_episodes = 1
    eplen = eval_py_env._eplen
    rlpost = np.empty([eplen,1], dtype=np.float32)
    factors = np.empty([eplen, eval_py_env._n_state], dtype=np.float32)
    for _ in range(num_episodes):
        time_step = eval_tf_env.reset()
        for i in range(eplen):
            action_step = policy.action(time_step)
            time_step = eval_tf_env.step(action_step.action)
            curr_obs = time_step.observation.numpy()[0]
            # RL agent action
            curraction_step = policy.action(time_step)
            curr_action = curraction_step.action[0].numpy()
            rlpost[i] = curr_action
            # Input factors
            factors[i,:] = curr_obs[-1,:]
        wealth = eval_py_env._traj_wealth
        stockret = eval_py_env._epdat_y
        mergekey = eval_py_env._epdat_key
    return (rlpost, factors, wealth, stockret, mergekey)

# def getSurprise(oldInfo, oldInfoSd, accInfo, accInfoSd):

#         # TASK compute the new information surprise (13 august 2012) 
#         usign = np.sign(accInfo - oldInfo)

#         # KL divergence between two normals (13 august 2012)
#         mu1 = oldInfo
#         sig1 = oldInfoSd
#         mu2 =  accInfo
#         sig2 = accInfoSd
#         kldiv = np.log(sig2/sig1) + ((sig1**2 + ((mu1-mu2)**2))/(2*(sig2**2))) - 0.5

#         surprise = usign*kldiv

#         return surprise


def getSurprise2(oldInfo, oldInfoSd, accInfo, accInfoSd):
        surprise = accInfo - oldInfo
        return surprise




def getRewardBias(surprise, theta, action, prevAction):
    surprise_magnitude = np.absolute(surprise)

    surprise_direction = np.sign(surprise)
    action_direction = np.sign(action-prevAction)
    isDirectionMatched = surprise_direction*action_direction

    rewardBias = isDirectionMatched * ((1+surprise_magnitude)**theta) * ((action - prevAction)**2)

    return rewardBias



def getBayesPrediction(signal_mean, signal_sd, prior_mean, prior_sd):

    # TASK   compute the Bayes posterior
    sig = signal_mean
    sigPrec = 1/(signal_sd**2)
    prior = prior_mean
    priorPrec = 1/(prior_sd**2)
    curr_bpost = (1/(sigPrec + priorPrec))*(sigPrec*sig + priorPrec*prior)
    
    return curr_bpost


def getSurpriseBayes(oldInfo, oldInfoSd, accInfo, accInfoSd, prior_mu, prior_sigma):
    oldBayes = getBayesPrediction(oldInfo, oldInfoSd, prior_mu, prior_sigma)
    newBayes = getBayesPrediction(accInfo, accInfoSd, prior_mu, prior_sigma)
    
    surprise = newBayes - oldBayes
    return surprise

def getSurprise(news, refBelief):
    surprise = news - refBelief
    return surprise

def getBetaFig(n_sim, eplen, policy, eval_tf_env, action_map):
    dat_sim = np.ndarray((eplen, 2, n_sim))
    for i in range(n_sim):
        rlpost, bayespost, fundamental = runEpisodes_basic_dqn(policy, eval_tf_env, action_map)
        dat_sim[:,0,i] = rlpost.reshape(eplen)
        dat_sim[:,1,i] = bayespost.reshape(eplen)

    beta = np.ndarray((eplen, 3))
    r_sq = np.ndarray((eplen))
    for i in range(eplen):
        x_rl = dat_sim[i,0,:]
        y_bayes = dat_sim[i,1,:]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_rl,y_bayes)
        r_sq[i]  = r_value**2
        beta[i,0] = slope
        beta[i,1] = slope + std_err
        beta[i,2] = slope - std_err
    return (beta, r_sq)

# Prep alpha data
def cleanDf_alpha(df, modelName):
    df = pd.read_csv(df)
    df["DATE"] = pd.to_datetime(df["DATE"])
    varli_fdmt = ["alpha"]
    if modelName == "sw":
        varli_fdmt = ["swalpha"]
    df = df[varli_fdmt + ["TICKER", "DATE"]]
    df = df.rename(columns={"TICKER":"tic", "DATE":"datadate"})
    df = df.sort_values(by=["tic", "datadate"])
    if modelName == "capm":
        df = df.rename(columns={"alpha":"alpha_capm"})
    elif modelName == "ff3":
        df = df.rename(columns={"alpha":"alpha_ff3"})
    elif modelName == "ff4":
        df = df.rename(columns={"alpha":"alpha_ff4"})
    elif modelName == "sw":
        df = df.rename(columns={"swalpha":"alpha_sw"})
    return df 

# Prep the beta suite (daily)
def cleanDf_betam(dfAdd_betam):
    df = pd.read_csv(dfAdd_betam)
    df["DATE"] = pd.to_datetime(df["DATE"])
    varli_fdmt = ["RET", "b_mkt", "alpha", "ivol", "exret"]
    df = df[varli_fdmt + ["TICKER", "DATE"]]
    df = df.rename(columns={"TICKER":"tic", "DATE":"datadate", "RET":"ret"})
    df['ret'] = df['ret'].str.rstrip('%').astype('float') / 100.0
    df['exret'] = df['exret'].str.rstrip('%').astype('float') / 100.0
    df['ivol'] = df['ivol'].str.rstrip('%').astype('float') / 100.0
    df = df.sort_values(by=["tic", "datadate"])
    return df 

# Prep Compustat (yearly)
def cleanDf_compustat(dfAdd_compustat):
    df = pd.read_csv(dfAdd_compustat)
    df["datadate"] = pd.to_datetime(df["datadate"])
    varli_fdmt = ["at", "capx", "ceq", "che", "csho", "dlc", "ebit", "invt", "revt", "xint", "xrd", "prcc_f"]
    df = df[varli_fdmt + ["tic", "datadate"]]
    df = df.sort_values(by=["tic", "datadate"]) 
    return df

# Prep CRSP (monthly)
def cleanDf_crsp_treasury(dfAdd_crsp_treasury):
    df = pd.read_csv(dfAdd_crsp_treasury)
    df["caldt"] = pd.to_datetime(df["caldt"])
    varli_fdmt = ["cpiret", "t90ret", "b2ret"]
    df = df[varli_fdmt + ["caldt"]]
    df = df.rename(columns={"caldt":"datadate"})
    df = df.sort_values(by=["datadate"]) 
    return df

def cleanDf_crsp_vwret(dfAdd_marektret):
    df = pd.read_csv(dfAdd_marektret)
    df["DATE"] = pd.to_datetime(df["DATE"])
    varli_fdmt = ["vwretd", "vwretx", "sprtrn"]
    df = df[varli_fdmt + ["DATE"]]
    df = df.rename(columns={"DATE":"datadate"})
    df = df.sort_values(by=["datadate"]) 
    return df

def mergeLowerFreqDf(df_high, df_low, leftDfName, rightDfName):
    if leftDfName=="betasuite" and rightDfName=="compustat":
        # df_high = df_betam
        # df_low = df_compustat
        mergedDf = pd.merge(df_high, df_low, 
                left_on = [df_high["tic"], df_high["datadate"].dt.year],
                right_on = [df_low["tic"], df_low["datadate"].dt.year],
                how = "left")
        mergedDf = mergedDf.dropna()
        mergedDf = mergedDf.drop(columns=["key_0", "key_1", "tic_y", "datadate_y"])
        mergedDf = mergedDf.rename(columns={"tic_x":"tic", "datadate_x":"datadate"})
        mergedDf = mergedDf.sort_values(by=["tic", "datadate"])
    elif leftDfName=="df_1" and rightDfName=="crsp_treasury":      
        # df_high = df_1
        # df_low = df_crsptreasury
        mergedDf = pd.merge(df_high, df_low, 
                        left_on = [df_high["datadate"].dt.year, df_high["datadate"].dt.month],
                        right_on = [df_low["datadate"].dt.year, df_low["datadate"].dt.month],
                        how = "left")
        mergedDf = mergedDf.dropna()
        mergedDf = mergedDf.drop(columns=["key_0", "key_1", "datadate_y"])
        mergedDf = mergedDf.rename(columns={"datadate_x":"datadate"})
        mergedDf = mergedDf.sort_values(by=["tic", "datadate"])
    elif leftDfName=="df_2" and rightDfName=="df_finratio":      
        # df_high = df_1
        # df_low = df_crsptreasury
        mergedDf = pd.merge(df_high, df_low, 
                        left_on = [df_high["tic"], df_high["datadate"].dt.year, df_high["datadate"].dt.month],
                        right_on = [df_low["tic"], df_low["datadate"].dt.year, df_low["datadate"].dt.month],
                        how = "left")
        mergedDf = mergedDf.dropna()
        mergedDf = mergedDf.drop(columns=["key_0", "key_1", "key_2", "tic_y", "datadate_y"])
        mergedDf = mergedDf.rename(columns={"tic_x":"tic", "datadate_x":"datadate"})
        mergedDf = mergedDf.sort_values(by=["tic", "datadate"])
        popped_tic = mergedDf.pop("tic")
        popped_datadate = mergedDf.pop("datadate")
        mergedDf.insert(0, 'datadate', popped_datadate)
        mergedDf.insert(0, 'tic', popped_tic)
        mergedDf = mergedDf.reset_index(drop=True)
    elif leftDfName=="df_3" and rightDfName=="df_marketret":
        mergedDf = pd.merge(df_high, df_low, 
                                left_on = [df_high["datadate"].dt.year, df_high["datadate"].dt.month, df_high["datadate"].dt.day],
                                right_on = [df_low["datadate"].dt.year, df_low["datadate"].dt.month, df_low["datadate"].dt.day],
                                how = "left", validate = "m:1")
        mergedDf = mergedDf.dropna()
        mergedDf = mergedDf.drop(columns=["key_0", "key_1", "key_2", "datadate_y"])
        mergedDf = mergedDf.rename(columns={"datadate_x":"datadate"})
        mergedDf = mergedDf.sort_values(by=["tic", "datadate"])
        popped_tic = mergedDf.pop("tic")
        popped_datadate = mergedDf.pop("datadate")
        mergedDf.insert(0, 'datadate', popped_datadate)
        mergedDf.insert(0, 'tic', popped_tic)
        mergedDf = mergedDf.reset_index(drop=True)
    return mergedDf
