# -*- coding: utf-8 -*-
"""
Training script for CVaR-QRDQN
"""

import numpy as np
import torch
from environment import create_environment
from agent import Agent

def train(env_name, num_episodes, max_steps, epsilon_start, epsilon_end, epsilon_decay, lr, gamma, cvar_alpha, num_quantiles):
    env = create_environment(env_name)
    agent = Agent(env.observation_space.shape[0], env.action_space.n, num_quantiles, lr, gamma, cvar_alpha)

    epsilon = epsilon_start
    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state, epsilon)
            next_state, reward, done, info, _ = env.step(action)
            total_reward += reward

            states, actions, rewards, next_states, dones = \
                np.array(state)[np.newaxis, :], np.array([action])[np.newaxis, :], np.array([reward])[np.newaxis, :], np.array(next_state)[np.newaxis, :], np.array([done])[np.newaxis, :]

            agent.update(batch=(states, actions, rewards, next_states, dones))

            state = next_state
            if done:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    # Save the model
    torch.save(agent.model.state_dict(), "cvar_qrdqn_model.pth")

if __name__ == "__main__":
    train(env_name="CartPole-v1", num_episodes=500, max_steps=200, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, lr=0.001, gamma=0.99, cvar_alpha=0.1, num_quantiles=50)