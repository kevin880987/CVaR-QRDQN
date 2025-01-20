# -*- coding: utf-8 -*-
"""
Environment setup for CVaR-QRDQN
"""

import gym

def create_environment(env_name):
    env = gym.make(env_name)
    return env