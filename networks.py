import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 

class Actor(nn.Module):
    """
    Network used to parametrize the policy of an agent.
    Uses 2 linear layers, the first with ReLU activation,
    the second with softmax.
    """
    
    def __init__(self, observation_space, action_space, hidden_dim=64):
        """
        Parameters
        ----------
        observation_space: int
            Number of flattened entries of the state
        action_space: int
            Number of (discrete) possible actions to take
        hidden_dim: int
            Size of the hidden layer
        """
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(observation_space, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, action_space)
    
    def forward(self, state):
        out = F.relu(self.linear1(state))
        probs = F.softmax(self.linear2(out), dim=1)
        return probs
    
class Critic(nn.Module):
    """
    Network used to parametrize the Critic of an Actor-Critic agent.
    Uses 2 linear layers, both with ReLU activation.
    Returns the value of a state.
    """
    
    def __init__(self, observation_space, hidden_dim=64):
        """
        Parameters
        ----------
        observation_space: int
            Number of flattened entries of the state
        hidden_dim: int
            Size of the hidden layer
        """
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(observation_space, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        out = F.relu(self.linear1(state))
        return F.relu(self.linear2(out)) 
        