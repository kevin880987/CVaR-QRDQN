# -*- coding: utf-8 -*-
"""
CVaR-QRDQN implementation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QRDQN(nn.Module):
    def __init__(self, state_dim, action_dim, num_quantiles):
        super(QRDQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim * num_quantiles)
        self.fc_base_quantile = nn.Linear(128, action_dim * 1)

        # Define quantile fractions
        # e.g., for num_quantiles=50, these range from 1/50 to 50/50
        self.taus = torch.linspace(0, 1, steps=num_quantiles + 1)[1:]

    def forward(self, states):
        x = torch.relu(self.fc1(states))
        x = torch.relu(self.fc2(x))
        delta_quantiles = self.fc3(x).view(-1, self.action_dim, self.num_quantiles)
        base_quantiles = self.fc_base_quantile(x).view(-1, self.action_dim, 1)
        quantiles = torch.softmax(base_quantiles, dim=-1) + torch.softmax(delta_quantiles, dim=-1).cumsum(dim=-1)
        return quantiles

