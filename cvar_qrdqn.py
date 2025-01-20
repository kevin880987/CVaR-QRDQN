# -*- coding: utf-8 -*-
"""
CVaR-QRDQN implementation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class CVaRQRDQN(nn.Module):
    def __init__(self, state_dim, action_dim, num_quantiles):
        super(CVaRQRDQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim * num_quantiles)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        quantiles = self.fc3(x).view(-1, self.action_dim, self.num_quantiles)
        return quantiles

    def get_q_values(self, state, action):
        quantiles = self.forward(state)
        action_quantiles = quantiles.gather(1, action.unsqueeze(-1).expand(-1, -1, self.num_quantiles))
        return action_quantiles.mean(dim=-1)