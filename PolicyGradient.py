import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical

from networks import Actor #custom module

class PolicyGrad():
    """
    Implements an RL agent with policy gradient method.
    
    Notes
    -----
    GPU implementation is just sketched; it works but it's slower than with CPU.
    """
    def __init__(self, observation_space, action_space, lr, gamma, discrete=False, project_dim = 4, device='cpu'):
        """
        Parameters
        ----------
        observation_space: int
            Number of flattened entries of the state
        action_space: int
            Number of (discrete) possible actions to take
        """
        
        self.gamma = gamma
        self.lr = lr
        
        self.n_actions = action_space
        self.discrete = discrete
        if self.discrete:
            self.net = Actor(observation_space, action_space, discrete, project_dim)
        else:
            self.net = Actor(observation_space, action_space, discrete)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        
        self.device = device 
        self.net.to(self.device) # move network to device
        
        
    def get_action(self, state, return_log=False):
        log_probs = self.forward(state)
        dist = torch.exp(log_probs)
       
        probs = Categorical(dist)
        action =  probs.sample().item()
        
        if return_log:
            return action, log_probs.view(-1)[action]
        else:
            return action
        
    def forward(self, state):
        if self.discrete:
            state = torch.from_numpy(state).to(self.device) 
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device) 
        return self.net(state)      
    
    def update(self, rewards, log_probs):
        
        ### Compute MC discounted returns ###
        
        Gamma = np.array([self.gamma**i for i in range(rewards.shape[0])])
        # reverse everything to use cumsum in right order, then reverse again
        Gt = np.cumsum(rewards[::-1]*Gamma[::-1])[::-1]
        # Rescale so that present reward is never discounted
        discounted_rewards =  Gt/Gamma
        
        dr = torch.tensor(discounted_rewards).to(self.device)
        dr = (dr - dr.mean())/dr.std()
        
        log_probs = torch.stack(log_probs)
        policy_gradient = -log_probs*dr
        policy_grad = policy_gradient.sum()
        
        self.optim.zero_grad()
        policy_grad.backward()
        self.optim.step()
        return policy_grad.item()
    
class PolicyGradEnt():
    """
    Implements an RL agent with policy gradient method.
    
    Notes
    -----
    GPU implementation is just sketched; it works but it's slower than with CPU.
    """
    def __init__(self, observation_space, action_space, lr, gamma, H, discrete=True, project_dim = 4, device='cpu'):
        """
        Parameters
        ----------
        observation_space: int
            Number of flattened entries of the state
        action_space: int
            Number of (discrete) possible actions to take
        """
        
        self.gamma = gamma
        self.lr = lr
        self.H = H # entropy coeff
        
        self.n_actions = action_space
        self.discrete = discrete
        if self.discrete:
            self.net = Actor(observation_space, action_space, discrete, project_dim)
        else:
            self.net = Actor(observation_space, action_space, discrete)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        
        self.device = device 
        self.net.to(self.device) # move network to device
        
        
    def get_action(self, state, return_log=False):
        log_probs = self.forward(state)
        dist = torch.exp(log_probs)
       
        probs = Categorical(dist)
        action =  probs.sample().item()
        
        if return_log:
            return action, log_probs.view(-1)[action], dist 
        else:
            return action
        
    def forward(self, state):
        if self.discrete:
            state = torch.from_numpy(state).to(self.device) 
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device) 
        return self.net(state)      
    
    def update(self, rewards, log_probs, distributions):
        
        ### Compute MC discounted returns ###
        
        Gamma = np.array([self.gamma**i for i in range(rewards.shape[0])])
        # reverse everything to use cumsum in right order, then reverse again
        Gt = np.cumsum(rewards[::-1]*Gamma[::-1])[::-1]
        # Rescale so that present reward is never discounted
        discounted_rewards =  Gt/Gamma
        
        dr = torch.tensor(discounted_rewards).to(self.device)
        dr = (dr - dr.mean())/dr.std()
        
        policy_gradient = []
        for log_prob, Gt in zip(log_probs, dr):
            policy_gradient.append(-log_prob*Gt) # "-" for minimization instead of maximization
           
        distributions = torch.stack(distributions).squeeze() # shape = (T,2)
        # Compute negative entropy (no - in front)
        entropy = torch.sum(distributions*torch.log(distributions), axis=1).sum()
        policy_grad = torch.stack(policy_gradient).sum()
        loss = policy_grad + self.H*entropy
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return policy_grad.item()