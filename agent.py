# -*- coding: utf-8 -*-
"""
Agent implementation for CVaR-QRDQN
"""

import numpy as np
import torch
import torch.optim as optim
from cvar_qrdqn import CVaRQRDQN

class Agent:
    def __init__(self, state_dim, action_dim, num_quantiles, lr, gamma, cvar_alpha):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.gamma = gamma
        self.cvar_alpha = cvar_alpha

        self.model = CVaRQRDQN(state_dim, action_dim, num_quantiles)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            quantiles = self.model(state)
            q_values = quantiles.mean(dim=-1)
            return q_values.argmax().item()

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        quantiles = self.model(states)
        next_quantiles = self.model(next_states).detach()

        q_values = quantiles.gather(1, actions.unsqueeze(-1).expand(-1, -1, self.num_quantiles)).mean(dim=-1)
        next_q_values = next_quantiles.mean(dim=-1).max(dim=1)[0]

        targets = rewards + self.gamma * next_q_values * (1 - dones)
        loss = (q_values - targets.unsqueeze(-1)).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()