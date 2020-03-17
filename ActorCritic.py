import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical

from networks import Actor, Critic, DiscreteActor, DiscreteCritic #custom module

class DiscreteA2C():
    """
    Implements Advantage Actor Critic RL agent. Uses episode trajectories to update.
    
    Notes
    -----
    GPU implementation is just sketched; it works but it's slower than with CPU.
    """
    
    def __init__(self, observation_space, action_space, lr, gamma, project_dim=8):
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
        self.actor = DiscreteActor(observation_space, action_space, project_dim)
        self.critic = DiscreteCritic(observation_space, project_dim)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
    def get_action(self, state, return_log=False, greedy=True):
        state = torch.from_numpy(state)
        log_probs = self.actor(state)
        dist = torch.exp(log_probs)
        probs = Categorical(dist)
        action =  probs.sample().item()
        if return_log:
            return action, log_probs.view(-1)[action]
        else:
            return action
    
    def update(self, rewards, log_probs, states, done):       
        ### Compute MC discounted returns ###
        # Discount factors
        Gamma = np.array([self.gamma**i for i in range(rewards.shape[0])])
        # reverse everything to use cumsum in right order, then reverse again
        Gt = np.cumsum(rewards[::-1]*Gamma[::-1])[::-1]
        # Rescale so that present reward is never discounted
        discounted_rewards =  Gt/Gamma
        
        # Wrap variables in tensors
        dr = torch.tensor(discounted_rewards).float()  
        old_states = torch.tensor(states[:-1]).view(1,-1)
        new_states = torch.tensor(states[1:]).view(1,-1)
        log_probs = torch.tensor(log_probs.astype(float)) 
        
        ### Update critic and then actor ###
        critic_loss = self.update_critic(dr, old_states)
        actor_loss = self.update_actor(dr, log_probs, new_states, old_states)
        return critic_loss, actor_loss
    
    def update_critic(self, dr, old_states):
        # Predictions
        V_pred = self.critic(old_states).squeeze()
        # MSE loss
        loss = torch.sum((V_pred - dr)**2)
        # backprop and update
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        return loss.item()
    
    def update_actor(self, dr, log_probs, new_states, old_states):
        # Get value as baseline
        V = self.critic(old_states).squeeze()
        # Compute advantage as total (discounted) return - value
        A = dr - V 
        # Rescale to unitary variance for a trajectory (axis=1)
        #A = (A - A.mean(axis=1).unsqueeze(1))/(A.std(axis=1).unsqueeze(1))
        #print("A ", A)
        #print("A ", A.shape)
        #print("log_probs ", log_probs.shape)
        # Compute - gradient
        policy_gradient = - log_probs*A
        #print("policy_gradient ", policy_gradient.shape)
        # Use it as loss
        policy_grad = torch.sum(policy_gradient)
        #print("policy_grad ", policy_grad)
        # barckprop and update
        self.actor_optim.zero_grad()
        policy_grad.backward()
        self.actor_optim.step()
        return policy_grad.item()
