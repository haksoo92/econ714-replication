from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

from collections import deque
import customutils
import statistics

tf.compat.v1.enable_v2_behavior()

class PriceInfoEnv(py_environment.PyEnvironment):

    def __init__(self, retdat):
        """
        Scripts
            (1) Draw a trading date from a training sample
            (2) Specify action and observation specs
            (3) Store transition relevant data
        """

        # Import the relevant data
        x_m3mo, x_p12mo = customutils.getEpisodePriceData(retdat)
        
        # Specify specs
        min_action, max_action = 0, 1 # portfolio weights (incorp short-sale const)        
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.float32, name='action', 
            minimum = min_action, 
            maximum = max_action)

        self._observation_spec = array_spec.ArraySpec(
            shape = (x_m3mo.shape), 
            dtype = np.float32, 
            name  = 'observation')

        self._state = x_m3mo
        self._epdat = x_p12mo
        self._t = 1
        self._episode_ended = False
        self._n_state = 4
        self._eplen = 252 # length of episode, 360 trading days
        self._traj_wealth = np.zeros(self._eplen + 1, )
        self._traj_wealth[0,] = 1 # initial wealth
        self._prevWealth = 1
        self._feedback_interval = 63
        self._rf = 1.0 # risk-free rate daily
        self._retdat = retdat

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        x_m3mo, x_p12mo = customutils.getEpisodePriceData(self._retdat)

        self._state = x_m3mo
        self._epdat = x_p12mo
        self._t = 1
        self._episode_ended = False
        self._n_state = 4
        self._eplen = 252 # length of episode, 360 trading days
        self._traj_wealth = np.zeros(self._eplen + 1, )
        self._traj_wealth[0,] = 1 # initial wealth
        self._prevWealth = 1
        self._feedback_interval = 63
        self._rf = 1.0 # risk-free rate daily
        
        return ts.restart(np.array(self._state, dtype=np.float32))

    def _step(self, action):
        """
        Compute the period reward if 90-day feedback interval.
        Update the state
            Fill in _traj_wealth
            Update _prevWealth
            Update _state
            Update _epsiode_ended
        """

        # Termination condition
        if self._episode_ended:
            return self.reset()


        # Stock return realization
        prevWealth = self._prevWealth
        # prevPrice = self._state[-1, 0] # closing price
        # currPrice = self._epdat[self._t - 1, 0] # closing price
        currReturn = self._epdat[self._t - 1, 0]
        currWealth = customutils.getCurrWealth(prevWealth, action, currReturn)
        self._traj_wealth[self._t] = currWealth
        self._prevWealth = currWealth


        # Compute rewards
        if self._t % self._feedback_interval == 0:
            evalPeriodWealthTraj = self._traj_wealth[self._t - self._feedback_interval + 1 : self._t + 1] # wealth trajectory in the previous evaluation interval
            reward = customutils.getSharpe(evalPeriodWealthTraj, self._rf) # daily sharpe over this period
        else:
            reward = 0.0


        # Update the state
        self._state = np.append(self._state[1:,:], self._epdat[self._t - 1, :].reshape((1,4)), axis = 0)
        

        # Update _t and _episode_ended flag
        if self._t == self._eplen: 
            self._episode_ended = True
        else:
            self._t += 1


        # Terminate or transition
        if self._episode_ended:
            return ts.termination(np.array(self._state, dtype=np.float32), reward)
        else:
            return ts.transition(np.array(self._state, dtype=np.float32), reward=reward, discount=1.0)
