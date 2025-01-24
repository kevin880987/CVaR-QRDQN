# -*- coding: utf-8 -*-
"""
Agent implementation for CVaR-QRDQN
"""

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
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

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))

        quantiles = self.model(states)
        next_quantiles = self.model(next_states).detach()

        q_values = self.model.get_q_values(states, actions)
        next_q_values = next_quantiles.mean(dim=-1).max(dim=1)[0]

        targets = rewards + self.gamma * next_q_values * (1 - dones)
        targets = targets.unsqueeze(-1).expand(-1, -1, self.num_quantiles)

        # Calculate Huber loss
        td_errors = quantiles - targets
        huber_loss = F.smooth_l1_loss(td_errors, torch.zeros_like(td_errors), reduction='none')

        # Calculate CVaR
        cvar_loss = huber_loss.mean(dim=-1)
        cvar_loss = cvar_loss.mean()

        self.optimizer.zero_grad()
        cvar_loss.backward()
        self.optimizer.step()