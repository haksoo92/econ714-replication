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

import rlprice
import customutils as cutil

tf.compat.v1.enable_v2_behavior()

# TEST SIMPLE LEARNING MODEL (LEARNING FROM PRIVATE SIGNAL)
# .............................................................................
# PARAMETERS
env_name = "rlprice" # @param {type:"string"}

# # number of updates to the model
# num_iterations = 10000 # @param {type:"integer"}, about 3 hours?

# initial_collect_steps = 50  # @param {type:"integer"} 
# # after each iteration, update the training model, so iteration re\fers to iteration of the update
# # steps per iteration refers to the number of steps a policy will take before getting updated
# # collect_steps_per_iteration = 8  # @param {type:"integer"}
# replay_buffer_max_length = 1000  # @param {type:"integer"}

# batch_size = 64  # @param {type:"integer"} the batch size of the sample tuples that are representative of the environment
# learning_rate = 1e-3  # @param {type:"number"}
# log_interval = 20  # @param {type:"integer"} the loss function of the model, assuming on the training?

# num_eval_episodes = 100  # @param {type:"integer"} the model evaluated over 10 episodes
# eval_interval = 100  # @param {type:"integer"} evaluation of the task of predicting fundmental

# # number of layers in the deep learning agent 
# fc_layer_params = (100,) # @param
# # the number of episodes before updating the model
# collect_episodes_per_iteration = 25 # @param
# # discretized action space sparseness, unused
# # n_discretize = 1000 # @param not used


# OUTPUT DIRECTORY 
policy_dir = './out/' + env_name + '/policy' 
returns_dir = './out/' + env_name + '/returns/returns.npy'
rlfolder_dir = './out/' + env_name
plotfolder_dir = './out/' + env_name + '/plot' 

# .............................................................................
# EPISODE ANALYSIS
infoLs = ["invt_act", "npm", "qqq", "ccc", "ivol", "ccap_equity", "ccap_debt"]
infoLsDisp = ["INVT", "NPM", "Q", "C", "IVOL", "CCAP-Equity", "CCAP-Debt"]

dfAdd_all = "/home/hsl612/gitWorkspace/priceinfo-rl/data/df_all.csv"
df_all = pd.read_csv(dfAdd_all)

# reload the model policy
policy = tf.compat.v2.saved_model.load(policy_dir)

eval_py_env = rlprice.PriceInfoEnv(df_all, infoLs)
eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)

eplen = eval_py_env._eplen
steps = range(eplen)
oneline = np.ones([eplen+1,1])
num_fig = 10
n_factors = len(infoLs)
for i in range(num_fig):
    epi_fig = plotfolder_dir + '/episode_' + str(i) + '.png'

    (rlpost, factors, wealth, stockret) = cutil.runEpisodesEmpirical(policy, eval_tf_env, eval_py_env)

    fig, axs = plt.subplots(2 + n_factors)
    axs[0].plot(steps, stockret, 'k--', label='Return Realized (Daily)', linewidth=1.0)
    axs[0].set_ylabel("Realized Returns (Daily)")
    axtwin = axs[0].twinx()
    axtwin.plot([-1] + [*steps], wealth, 'r-', label='Wealth')
    axtwin.set_ylabel("Wealth")
    axtwin.plot([-1] + [*steps], np.ones([eplen+1,1]), 'k-', linewidth=1.5)
    # axs[0].plot(steps, rlpost, 'r-', label='RL')
    plt.setp(axs[0], xlabel='Step in Episode')
    # plt.setp(axs[0], ylabel='Posterior')
    axs[0].legend(loc='lower right')
    axtwin.legend(loc='lower left')

    axs[1].plot(steps, rlpost, 'b-', label='RL Portfolio Weight')
    plt.setp(axs[1], xlabel='Step in Episode')
    plt.setp(axs[1], ylabel='RL Portfolio Weight')
    axs[1].legend(loc='lower right')

    for i in range(n_factors):
        axs[2+i].plot(steps, factors[:,i], 'b--', label=infoLsDisp[i])
        plt.setp(axs[2+i], xlabel='Step in Episode')
        plt.setp(axs[2+i], ylabel=infoLsDisp[i])
        axs[2+i].legend(loc='lower right')

    fig.set_size_inches(12, 6 + 3*n_factors)
    plt.ylim()
    # plt.show()
    # plt.close()
    plt.savefig(epi_fig)
    plt.close()


# .............................................................................
# TEST FOR CONVERGENCE

plt.clf()
fig, axs = plt.subplots(1)
returns_fig = plotfolder_dir + '/returns.png'
with open(returns_dir, 'rb') as f:
    returns = np.load(f)
steps = range(0, num_iterations + 1, eval_interval)
axs.plot(steps[1:-1], returns[1:-1], 'k')
plt.setp(axs, xlabel='Iteration')
plt.setp(axs, ylabel='Average Return')

fig.set_size_inches(12, 5)
plt.ylim()
plt.savefig(returns_fig)
plt.close()