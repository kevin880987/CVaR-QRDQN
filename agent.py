# -*- coding: utf-8 -*-
"""
Agent implementation for CVaR-QRDQN
"""
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from cvar_qrdqn import QRDQN

class Agent:
    def __init__(self, state_dim, action_dim, num_quantiles, lr, gamma, cvar_alpha):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.gamma = gamma
        self.cvar_alpha = cvar_alpha

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = QRDQN(state_dim, action_dim, num_quantiles).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_q_values(self, states, actions):
        quantiles = self.model(states.to(self.device))
        cvar_quantile = max(1, int(self.cvar_alpha*self.num_quantiles)+1)
        action_quantiles = quantiles[:, :, :cvar_quantile].mean(dim=-1).gather(1, actions.to(self.device))
        return action_quantiles.to('cpu')

    def select_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            quantiles = self.model(state)
            cvar_quantile = max(1, int(self.cvar_alpha*self.num_quantiles)+1)
            q_values = quantiles[:, :, :cvar_quantile].mean(dim=-1)
            return q_values.to('cpu').argmax().item()

    def select_actions(self, states, epsilon=0.1):
        return torch.tensor([self.select_action(state, epsilon) for state in states]).unsqueeze(-1)
        
    def update(self, batch):
        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        quantiles = self.model(states)
        next_quantiles = self.model(next_states)

        targets = rewards + self.gamma * next_quantiles * (1 - dones)
        targets = targets.expand(-1, self.action_dim, self.num_quantiles)
        td_errors = quantiles - targets
        huber_loss = F.smooth_l1_loss(td_errors, torch.zeros_like(td_errors), reduction='none')

        # targets = rewards + self.gamma * self.get_q_values(next_states, self.select_actions(next_states)) * (1 - dones)
        # targets = targets.unsqueeze(-1).expand(-1, self.num_quantiles)

        # td_errors = self.get_q_values(states, actions) - targets
        # huber_loss = F.smooth_l1_loss(td_errors, torch.zeros_like(td_errors), reduction='none')

        # Use tau to weight the loss
        tau = self.model.taus.unsqueeze(0).unsqueeze(0)
        huber_loss = (tau - (td_errors < 0).float()).abs() * huber_loss
        huber_loss = huber_loss.mean(dim=-1).mean()

        self.optimizer.zero_grad()
        huber_loss.backward()
        self.optimizer.step()