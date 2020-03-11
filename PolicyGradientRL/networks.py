import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
    
class Actor(nn.Module):
    """
    Network used to parametrize the policy of an agent.
    Uses 3 linear layers, the first 2 with ReLU activation,
    the third with softmax.
    """
    
    def __init__(self, observation_space, action_space):
        """
        Parameters
        ----------
        observation_space: int
            Number of flattened entries of the state
        action_space: int
            Number of (discrete) possible actions to take
        """
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(observation_space, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, action_space)
    
    def forward(self, state):
        out = F.relu(self.linear1(state))
        out = F.relu(self.linear2(out))
        probs = F.softmax(self.linear3(out), dim=1)
        return probs
    
class Critic(nn.Module):
    """
    Network used to parametrize the Critic of an Actor-Critic agent.
    Uses 3 linear layers, only the first 2 with ReLU activation.
    Returns the value of a state.
    """
    
    def __init__(self, observation_space):
        """
        Parameters
        ----------
        observation_space: int
            Number of flattened entries of the state
        hidden_dim: int
            Size of the hidden layer
        """
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(observation_space, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 1)
    
    def forward(self, state):
        out = F.relu(self.linear1(state))
        out = F.relu(self.linear2(out))
        return self.linear3(out) 
        