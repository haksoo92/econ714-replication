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

from tf_agents.environments.customenv import rlfund as rlprice
import customutils as cutil
import pickle
import json

tf.compat.v1.enable_v2_behavior()

# .............................................................................
# SET PARAMETERS
env_name = "rlprice"

policy_dir = './out/' + env_name + '/policy' 
returns_dir = './out/' + env_name + '/returns/returns.npy'

dur_iter = 60*60*5 # 5 hours
# dur_iter = 300

initial_collect_steps = 1000
collect_steps_per_iteration = 4
replay_buffer_max_length = 100000

batch_size = 16
learning_rate = 1e-4

num_eval_episodes = 10
eval_interval = 100
log_interval = 100

num_actions = 50 # 2% fineness action space [0,1]

fc_layer_params = (100,)
collect_episodes_per_iteration = 4

infoLs = ["prccd", "prchd", "prcld", "prcod"]
dfAdd_all = "/home/hsl612/gitWorkspace/priceinfo-rl/data/df_all.csv"
df_all = pd.read_csv(dfAdd_all)


# .............................................................................
# SAVE HYPERPARAMS
outdict = {"env_name":env_name, "policy_dir":policy_dir, "returns_dir":returns_dir, "dur_iter":dur_iter, "initial_collect_steps":initial_collect_steps, "collect_steps_per_iteration":collect_episodes_per_iteration, "replay_buffer_max_length":replay_buffer_max_length, "batch_size":batch_size, "learning_rate":learning_rate, "num_eval_episodes":num_eval_episodes, "eval_interval":eval_interval, "log_interval":log_interval, "num_actions":num_actions, "fc_layer_params":fc_layer_params, "collect_episodes_per_iteration":collect_episodes_per_iteration, "infoLs":infoLs, "dfAdd_all":dfAdd_all}
hyperparams_json = json.dumps(outdict)
f = open(policy_dir + "/hyperparams.json", "w")
f.write(hyperparams_json)
f.close

# .............................................................................
# TEST BASIC ENVIRONMENT FUNCTIONS
# env = rlprice.PriceInfoEnv(df_all, infoLs)
# env.reset()

# print('Observation Spec:')
# print(env.time_step_spec().observation)

# print('Reward Spec:')
# print(env.time_step_spec().reward)

# print('Action Spec:')
# print(env.action_spec())

# time_step = env.reset()
# print('Time step:')
# print(time_step)

# action = np.array(1, dtype=np.int32)

# next_time_step = env.step(action)
# print('Next time step:')
# print(next_time_step)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# MODEL TRAINING

train_py_env = rlprice.PriceInfoEnv(df_all, infoLs)
eval_py_env = rlprice.PriceInfoEnv(df_all, infoLs)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.compat.v2.Variable(0)

# global_step = tf.compat.v1.train.get_or_create_global_step()
tf_agent = reinforce_agent.ReinforceAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_network=actor_net,
    optimizer=optimizer,
    normalize_returns=True,
    train_step_counter=train_step_counter)
tf_agent.initialize()

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = environment.step(action_step.action)
        episode_return += time_step.reward
    total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)


def collect_episode(environment, policy, num_episodes):
    episode_counter = 0
    environment.reset()
    while episode_counter < num_episodes:
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)
        
        if traj.is_boundary():
            episode_counter += 1


# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
returns = [avg_return]

then = datetime.now()
# for _ in range(num_iterations):
while(True):
    # Collect a few episodes using collect_policy and save to the replay buffer.
    collect_episode(train_env, tf_agent.collect_policy, collect_episodes_per_iteration)

    # Use data from the buffer and update the agent's network.
    experience = replay_buffer.gather_all()
    train_loss = tf_agent.train(experience)
    replay_buffer.clear()
    step = tf_agent.train_step_counter.numpy()

    if step % log_interval == 0:
        now = datetime.now()
        duration = now - then
        duration_sec = duration.total_seconds()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))
        print("Seconds until training completion =", dur_iter - duration_sec)
        if duration_sec > dur_iter: # termination condition
            break
    
    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)

# save the model policy
tf_policy_saver = policy_saver.PolicySaver(tf_agent.policy)
tf_policy_saver.save(policy_dir)

# reload the model policy
saved_policy = tf.compat.v2.saved_model.load(policy_dir)

# save returns in epsisode averaged
with open(returns_dir, 'wb') as f:
    np.save(f, returns)
with open(returns_dir, 'rb') as f:
    returns = np.load(f)
