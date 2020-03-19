import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical

from networks import Actor, DiscreteActor #custom module

class PolicyGrad():
    """
    Implements an RL agent with policy gradient method.
    
    Notes
    -----
    GPU implementation is just sketched; it works but it's slower than with CPU.
    """
    def __init__(self, observation_space, action_space, lr, gamma, device='cpu'):
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
        self.net = Actor(observation_space, action_space)
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
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device) 
        return self.net(state)      
    
    def update(self, rewards, log_probs):
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt += (self.gamma**pw)*r
                pw += 1 # exponent of the discount
            discounted_rewards.append(Gt)
        
        dr = torch.tensor(discounted_rewards).to(self.device)
        dr = (dr - dr.mean())/dr.std()
        
        policy_gradient = []
        log_probs = torch.stack(log_probs)
        #for log_prob, Gt in zip(log_probs, dr):
        #    policy_gradient.append(-log_prob*Gt) # "-" for minimization instead of maximization
        policy_gradient = -log_probs*dr
        self.optim.zero_grad()
        #policy_grad = torch.stack(policy_gradient).sum()
        policy_grad = policy_gradient.sum()
        #print("policy_grad ", policy_grad)
        policy_grad.backward()
        self.optim.step()
        return policy_grad.item()

class DiscretePolicyGrad():
    """
    Implements an RL agent with policy gradient method.
    
    Notes
    -----
    GPU implementation is just sketched; it works but it's slower than with CPU.
    """
    def __init__(self, observation_space, action_space, lr, gamma, project_dim=16, device='cpu'):
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
        self.net = DiscreteActor(observation_space, action_space, project_dim)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        
        #self.device = device 
        #self.net.to(self.device) # move network to device
        
        
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
        state = torch.from_numpy(state)#.to(self.device) 
        return self.net(state)      
    
    def update(self, rewards, log_probs):
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt += (self.gamma**pw)*r
                pw += 1 # exponent of the discount
            discounted_rewards.append(Gt)
        dr = torch.tensor(discounted_rewards)#.to(self.device)
        dr = (dr - dr.mean())/(dr.std()+1e-4)
        #print("dr ", dr)
        policy_gradient = []
        for log_prob, Gt in zip(log_probs, dr):
            policy_gradient.append(-log_prob*Gt) # "-" for minimization instead of maximization
        #print("Gradients (before zero grad): ")
        #for p in self.net.parameters():
        #    print(p.grad)
        self.optim.zero_grad()
        #print("Gradients (after zero grad): ")
        #for p in self.net.parameters():
        #    print(p.grad)
        #print("polcy gradient ", policy_gradient)
        policy_grad = torch.stack(policy_gradient).sum()
        #print("policy_grad ", policy_grad)
        policy_grad.backward()
        #print("Gradients (after backward): ")
        #for p in self.net.parameters():
        #    print(p.shape)
        #    print(p.grad)
        self.optim.step()
        return policy_grad.item()
    
class PolicyGradEnt():
    """
    Implements an RL agent with policy gradient method.
    
    Notes
    -----
    GPU implementation is just sketched; it works but it's slower than with CPU.
    """
    def __init__(self, observation_space, action_space, lr, gamma, H, device='cpu'):
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
        self.net = Actor(observation_space, action_space)
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
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device) 
        return self.net(state)      
    
    def update(self, rewards, log_probs, distributions):
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt += (self.gamma**pw)*r
                pw += 1 # exponent of the discount
            discounted_rewards.append(Gt)
        
        dr = torch.tensor(discounted_rewards).to(self.device)
        dr = (dr - dr.mean())/dr.std()
        
        policy_gradient = []
        for log_prob, Gt in zip(log_probs, dr):
            policy_gradient.append(-log_prob*Gt) # "-" for minimization instead of maximization
           
        distributions = torch.stack(distributions).squeeze() # shape = (T,2)
        #print("distributions ", distributions.shape)
        #print("distributions ", distributions)
        # Compute negative entropy (no - in front)
        entropy = torch.sum(distributions*torch.log(distributions), axis=1).sum()
        #print("H x entropy ", self.H*entropy)
        policy_grad = torch.stack(policy_gradient).sum()
        #print("Policy grad ", policy_grad)
        loss = policy_grad + self.H*entropy
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return policy_grad.item()

class DiscretePolicyGradEnt():
    """
    Implements an RL agent with policy gradient method.
    
    Notes
    -----
    GPU implementation is just sketched; it works but it's slower than with CPU.
    """
    def __init__(self, observation_space, action_space, lr, gamma, H, project_dim=4, device='cpu'):
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
        self.net = DiscreteActor(observation_space, action_space, project_dim)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        
        #self.device = device 
        #self.net.to(self.device) # move network to device
        
        
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
        state = torch.from_numpy(state)#.float().unsqueeze(0).to(self.device) 
        return self.net(state)      
    
    def update(self, rewards, log_probs, distributions):
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt += (self.gamma**pw)*r
                pw += 1 # exponent of the discount
            discounted_rewards.append(Gt)
        
        dr = torch.tensor(discounted_rewards)#.to(self.device)
        dr = (dr - dr.mean())/(dr.std()+1e-4)
        
        policy_gradient = []
        for log_prob, Gt in zip(log_probs, dr):
            policy_gradient.append(-log_prob*Gt) # "-" for minimization instead of maximization
           
        distributions = torch.stack(distributions).squeeze() # shape = (T,2)
        #print("distributions ", distributions.shape)
        #print("distributions ", distributions)
        # Compute negative entropy (no - in front)
        entropy = torch.sum(distributions*torch.log(distributions), axis=1).sum()
        #print("H x entropy ", self.H*entropy)
        policy_grad = torch.stack(policy_gradient).sum()
        #print("Policy grad ", policy_grad)
        loss = policy_grad + self.H*entropy
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return policy_grad.item()