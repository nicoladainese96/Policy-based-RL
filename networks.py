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
    
    def __init__(self, observation_space, action_space, discrete=False, project_dim=4):
        """
        Parameters
        ----------
        observation_space: int
            Number of flattened entries of the state
        action_space: int
            Number of (discrete) possible actions to take
        discrete: bool
            If True, adds an embedding layer before the linear layers
        project_dim: int
            Dimension of the embedding space
        """
        super(Actor, self).__init__()
        self.discrete = discrete
        if self.discrete:
            self.embedding = nn.Embedding(observation_space, project_dim)
            self.linear1 = nn.Linear(project_dim, 64)
        else:
            self.linear1 = nn.Linear(observation_space, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, action_space)
    
    def forward(self, state):
        if self.discrete:
            state = self.embedding(state)
        out = F.relu(self.linear1(state))
        out = F.relu(self.linear2(out))
        log_probs = F.log_softmax(self.linear3(out), dim=1)
        return log_probs
        
class BasicCritic(nn.Module):
    """
    Network used to parametrize the Critic of an Actor-Critic agent.
    Uses 3 linear layers, only the first 2 with ReLU activation.
    Returns the value of a state.
    """
    
    def __init__(self, observation_space, discrete=False, project_dim=4):
        """
        Parameters
        ----------
        observation_space: int
            Number of flattened entries of the state
        hidden_dim: int
            Size of the hidden layer
        """
        super(BasicCritic, self).__init__()
        self.discrete = discrete
        if self.discrete:
            self.embedding = nn.Embedding(observation_space, project_dim)
            self.linear1 = nn.Linear(project_dim, 64)
        else:
            self.linear1 = nn.Linear(observation_space, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 1)
    
    def forward(self, state):
        if self.discrete:
            state = self.embedding(state)
        out = F.relu(self.linear1(state))
        out = F.relu(self.linear2(out))
        return self.linear3(out) 
    
class Critic(nn.Module):
    """Implements a generic critic, that can have 2 independent networks is twin=True. """
    def __init__(self, observation_space, discrete=False, project_dim=4, twin=False, target=False):
        super(Critic, self).__init__()
        
        self.twin = twin
        self.target = target
        
        if twin:
            self.net1 = BasicCritic(observation_space, discrete, project_dim)
            self.net2 = BasicCritic(observation_space, discrete, project_dim)
        else:
            self.net = BasicCritic(observation_space, discrete, project_dim)
        
    def forward(self, state):
        if self.twin:
            v1 = self.net1(state)
            v2 = self.net2(state)
            if self.target:
                v = torch.min(v1, v2) # one could also try with the mean, for a less unbiased estimate
            else:
                return v1, v2
        else:
            v = self.net(state)
            
        return v
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    