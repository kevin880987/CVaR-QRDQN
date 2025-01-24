# -*- coding: utf-8 -*-
"""
Main script to execute CVaR-QRDQN demo
"""

from train import train
from evaluate import evaluate

def main():
    # Training parameters
    env_name = "CartPole-v1"
    num_episodes = 500
    max_steps = 200
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    lr = 0.001
    gamma = 0.99
    cvar_alpha = 0.1
    num_quantiles = 50

    # Train the model
    print("Training the CVaR-QRDQN model...")
    train(env_name, num_episodes, max_steps, epsilon_start, epsilon_end, epsilon_decay, lr, gamma, cvar_alpha, num_quantiles)

    # Evaluate the model
    model_path = "cvar_qrdqn_model.pth"
    num_eval_episodes = 10
    print("Evaluating the CVaR-QRDQN model...")
    evaluate(env_name, model_path, num_eval_episodes, max_steps, num_quantiles)

if __name__ == "__main__":
    main()