import multiprocessing as mp
# import main_rlfund
# import main_rlprice
# import main_rlagg
# import analysis_rltheo2 as rltheo_analysis
import main_rltheo as rltheo
import analysis_rltheo2 as rltheo_analysis
# import main_rltheo as rltheo

# ==================================================================================
# SET PARAMS
# durIter = 60*60*5
# evalInterval = 100
# logInterval = 200
durIter = 60*30
evalInterval = 1
logInterval = 1
# evalInterval = 1000
# logInterval = 1000

# ===================================================================================

dict_title = {"rltheo_sq":"Performance of the RL-Agent with the Squared-Error Rewards",
"rltheo_abs":"Performance of the RL-Agent with the Absolute-Error Rewards",
"rltheo_crra":"Performance of the RL-Agent with the CRRA Rewards",
"rltheo_crrahigh":"Performance of the RL-Agent with the Patient CRRA Rewards",
"rltheo_cara":"Performance of the RL-Agent with the CARA Rewards",
"rltheo_carahigh":"Performance of the RL-Agent with the Patient CARA Rewards",
"rltheo_sharpe":"Performance of the RL-Agent with the Sharpe Rewards",
"rltheo_expret":"Performance of the RL-Agent with the Expected Return Rewards"
}
# dict_title = {"rltheo_sq":"Performance of the RL-Agent with the Squared-Error Rewards",
# "rltheo_sharpe":"Performance of the RL-Agent with the Sharpe Rewards"
# }

dict_rewards = {"rltheo_sq":"direct-squared-loss",
"rltheo_abs":"direct-absolute-loss",
"rltheo_crra":"crra",
"rltheo_crrahigh":"crra-high",
"rltheo_cara":"cara",
"rltheo_carahigh":"cara-high",
"rltheo_sharpe":"sharpe",
"rltheo_expret":"expected-return"
}
# dict_rewards = {"rltheo_sq":"direct-squared-loss",
# "rltheo_sharpe":"sharpe"
# }

# for key, value in dict_title.items():
#     currFolderName = "rltheo_noprice/" + key
#     rltheo.trainRlTheo(
#         folderName = currFolderName, 
#         dur_iter = durIter, 
#         eval_interval = evalInterval, 
#         log_interval = logInterval, num_actions = 50, rewardtype = dict_rewards[key], feedback_interval = 10, titleInput=value, isFundamentalOnly = True, isDQN = False)
#     rltheo_analysis.makePlotsRlTheo(folderName = currFolderName)

# for key, value in dict_title.items():
#     currFolderName = "rltheo_price/" + key
#     rltheo.trainRlTheo(
#         folderName = currFolderName, 
#         dur_iter = durIter, 
#         eval_interval = evalInterval, 
#         log_interval = logInterval, num_actions = 50, rewardtype = dict_rewards[key], feedback_interval = 10, titleInput=value, isFundamentalOnly = False, isDQN = False)
#     rltheo_analysis.makePlotsRlTheo(folderName = currFolderName)

def runPriceModels(key):
    currtitle = dict_title[key]
    num_action = 200
    feedback_interval = 63

    currFolderName = "rltheo_price/" + key
    rltheo.trainRlTheo(
        folderName = currFolderName, 
        dur_iter = durIter, 
        eval_interval = evalInterval, 
        log_interval = logInterval, num_actions = num_action, rewardtype = dict_rewards[key], feedback_interval = feedback_interval, titleInput=currtitle, isFundamentalOnly = False, isDQN = False)
    rltheo_analysis.makePlotsRlTheo(folderName = currFolderName)

    rltheo_analysis.makePlotsRlTheoCfRLs(isFundamentalOnly = False)
    rltheo_analysis.collectCompmat(isFundamentalOnly=False)

def runNopriceModels(key):
    currtitle = dict_title[key]
    num_action = 400
    feedback_interval = 63

    currFolderName = "rltheo_noprice/" + key
    rltheo.trainRlTheo(
        folderName = currFolderName, 
        dur_iter = durIter, 
        eval_interval = evalInterval, 
        log_interval = logInterval, num_actions = num_action, rewardtype = dict_rewards[key], feedback_interval = feedback_interval, titleInput=currtitle, isFundamentalOnly = True, isDQN = False)
    rltheo_analysis.makePlotsRlTheo(folderName = currFolderName)

    rltheo_analysis.makePlotsRlTheoCfRLs(isFundamentalOnly = True)
    rltheo_analysis.collectCompmat(isFundamentalOnly=True)

# ================================================================================
# START
# runls = ["rltheo_sq", "rltheo_sharpe", "rltheo_crra", "rltheo_expret", "rltheo_cara", "rltheo_abs", "rltheo_crrahigh", "rltheo_carahigh"]
runls = ["rltheo_sq", "rltheo_sharpe", "rltheo_crra", "rltheo_expret", "rltheo_cara", "rltheo_abs", "rltheo_crrahigh", "rltheo_carahigh"]

with mp.Pool(processes = 8) as p:
    results = p.map(runPriceModels, runls)
    results = p.map(runNopriceModels, runls)

# ================================================================================
# COMBINE

# rltheo_analysis.makePlotsRlTheoCfRLs(isFundamentalOnly = True)
# rltheo_analysis.collectCompmat(isFundamentalOnly=True)

# rltheo_analysis.makePlotsRlTheoCfRLs(isFundamentalOnly = False)
# rltheo_analysis.collectCompmat(isFundamentalOnly=False)

# ================================================================================








# MEMO

# TODO [FIGURE]
# Example episode given input plot different model outputs (differnt colors for post; price paths)

# TODO [TABLE]
#                   |  MSEreB(RLmodel-2) 
# MSEreB(RLmodel-1) |  MSD(RL1, RL2)     ...
# ...
# ...               |


# TODO Gray out RL posterior in pre episode.

# TODO Reagrange plots; change the name to epochs for x-axis


# runPriceModels("rltheo_sharpe")
# runNopriceModels("rltheo_sharpe")

# currFolderName = "rltheo_price/rltheo_sharpe"
# rltheo_analysis.makePlotsRlTheo(folderName = currFolderName)


# durIter = 60*0.01

# key = "rltheo_sharpe"
# currtitle = dict_title[key]
# num_action = 400
# feedback_interval = 63

# currFolderName = "rltheo_noprice/" + key
# rltheo.trainRlTheo(
#         folderName = currFolderName, 
#         dur_iter = durIter, 
#         eval_interval = evalInterval, 
#         log_interval = logInterval, num_actions = num_action, rewardtype = dict_rewards[key], feedback_interval = feedback_interval, titleInput=currtitle, isFundamentalOnly = False, isDQN = False)
# rltheo_analysis.makePlotsRlTheo(folderName = currFolderName)

# rltheo_analysis.makePlotsRlTheoCfRLs(isFundamentalOnly = False)

# curr = "rltheo_noprice/"
# rltheo.trainRlTheo(folderName = curr + "rltheo_sq_1", dur_iter = 60*1, eval_interval = 100, log_interval = 20, num_actions = 50, rewardtype = "direct-squared-loss", feedback_interval = 1, titleInput="SE", isFundamentalOnly=True)
# rltheo_analysis.makePlotsRlTheo(folderName = curr + "rltheo_sq_1")


# rltheo.trainRlTheo(folderName = "rltheo_abs_1", dur_iter = 60*30, eval_interval = 100, log_interval = 200, num_actions = 50, rewardtype = "direct-absolute-loss", feedback_interval = 1, titleInput="AE")
# rltheo_analysis.makePlotsRlTheo(folderName = "rltheo_abs_1")


# rltheo.trainRlTheo(folderName = "rltheo_crra_1", dur_iter = 60*1, eval_interval = 100, log_interval = 20, num_actions = 50, rewardtype = "crra", feedback_interval = 1, titleInput="AAA")
# rltheo_analysis.makePlotsRlTheo(folderName = "rltheo_crra_1")


# rltheo.trainRlTheo(folderName = "rltheo_crrahigh_1", dur_iter = 60*1, eval_interval = 100, log_interval = 20, num_actions = 50, rewardtype = "crra-high", feedback_interval = 1, titleInput=currTitle)
# rltheo_analysis.makePlotsRlTheo(folderName = "rltheo_crrahigh_1")


# rltheo.trainRlTheo(folderName = "rltheo_cara_1", dur_iter = 60*1, eval_interval = 100, log_interval = 20, num_actions = 50, rewardtype = "cara", feedback_interval = 1, titleInput=currTitle)
# rltheo_analysis.makePlotsRlTheo(folderName = "rltheo_cara_1")


# rltheo.trainRlTheo(folderName = "rltheo_carahigh_1", dur_iter = 60*1, eval_interval = 100, log_interval = 20, num_actions = 50, rewardtype = "cara-high", feedback_interval = 1, titleInput="Performance of the RL-Agent with the Patient CARA Rewards at 1-day Frequency")
# rltheo_analysis.makePlotsRlTheo(folderName = "rltheo_carahigh_1")


# rltheo.trainRlTheo(folderName = "rltheo_sharpe_63", dur_iter = 60*1, eval_interval = 100, log_interval = 10, num_actions = 50, rewardtype = "sharpe", feedback_interval = 63, titleInput="Performance of the RL-Agent with the Sharpe Rewards at 1-day Frequency")
# rltheo_analysis.makePlotsRlTheo(folderName = "rltheo_sharpe_63")


# rltheo.trainRlTheo(folderName = "rltheo_expret_63", dur_iter = 60*1, eval_interval = 100, log_interval = 10, num_actions = 50, rewardtype = "expected-return", feedback_interval = 63, titleInput="Performance of the RL-Agent with the Expected Return Rewards at 1-day Frequency")
# rltheo_analysis.makePlotsRlTheo(folderName = "rltheo_expret_63")



