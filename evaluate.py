# -*- coding: utf-8 -*-
"""
Evaluation script for CVaR-QRDQN
"""

import torch
from environment import create_environment
from agent import Agent

def evaluate(env_name, model_path, num_episodes, max_steps, num_quantiles):
    env = create_environment(env_name)
    agent = Agent(env.observation_space.shape[0], env.action_space.n, num_quantiles, lr=0, gamma=0, cvar_alpha=0)
    agent.model.load_state_dict(torch.load(model_path))

    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state, epsilon=0)
            next_state, reward, done, info, _ = env.step(action)
            total_reward += reward
            state = next_state
            if done:
                break

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

if __name__ == "__main__":
    evaluate(env_name="CartPole-v1", model_path="cvar_qrdqn_model.pth", num_episodes=10, max_steps=200, num_quantiles=50)