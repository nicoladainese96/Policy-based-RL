import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical

from networks import Actor, Critic, DiscreteActor, DiscreteCritic #custom module



    
class A2C_v1():
    """
    Implements Advantage Actor Critic RL agent. Uses episode trajectories to update.
    
    Notes
    -----
    GPU implementation is just sketched; it works but it's slower than with CPU.
    """
    
    def __init__(self, observation_space, action_space, lr, gamma, 
                 device='cpu', discrete=False, project_dim=8):
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
            self.actor = DiscreteActor(observation_space, action_space, project_dim)
            self.critic = DiscreteCritic(observation_space, project_dim)
        else:
            self.actor = Actor(observation_space, action_space)
            self.critic = Critic(observation_space)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        self.device = device 
        ### Not implemented ###
        #self.actor.to(self.device) # move network to device
        #self.critic.to(self.device)
        
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
            state = torch.from_numpy(state)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0) 
        log_probs = self.actor(state)
        return log_probs
    
    def update(self, rewards, log_probs, states, done):
        # Wrap variables in tensors
        if self.discrete:
            old_states = torch.tensor(states[:,:-1])
            new_states = torch.tensor(states[:,1:])
        else:
            old_states = torch.tensor(states[:,:-1]).float()
            new_states = torch.tensor(states[:,1:]).float()
        done = torch.LongTensor(done.astype(int))
        #log_probs = torch.tensor(log_probs.astype(float)) ### ERROR HERE
        log_probs = torch.stack(log_probs)
        # Update critic and then actor
        self.update_critic(rewards, new_states, old_states, done)
        self.update_actor(rewards, log_probs, new_states, old_states)
        return
    
    def update_critic(self, rewards, new_states, old_states, done):
        """
        Minimize \sum_{t=0}^{T-1}(rewards[t] + gamma V(new_states[t]) - V(old_states[t]) )**2
        where V(state) is the prediction of the critic.
        
        Parameters
        ----------
        reward: shape (T,)
        old_states, new_states: shape (T, observation_space)
        """
        rewards = torch.tensor(rewards)    #.to(self.device)
        #print("rewards.shape ", rewards.shape)
        # Predictions
        V_pred = self.critic(old_states).squeeze()
        #print("V_pred.shape ", V_pred.shape)
        # Targets
        V_trg = self.critic(new_states).squeeze().detach()
        #print("V_trg.shape ", V_trg.shape)
        V_trg = (1-done)*self.gamma*V_trg + rewards
        #print("V_trg.shape ", V_trg.shape)
        # MSE loss
        loss = torch.sum((V_pred - V_trg)**2)
        # backprop and update
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        return
    
    def update_actor(self, rewards, log_probs, new_states, old_states):
        # Discount factors
        Gamma = np.array([self.gamma**i for i in range(rewards.shape[1])]).reshape(1,-1)
        # reverse everything to use cumsum in right order, then reverse again
        Gt = np.cumsum(rewards[:,::-1]*Gamma[:,::-1], axis=1)[:,::-1]
        # Rescale so that present reward is never discounted
        discounted_rewards =  Gt/Gamma
        # Wrap into tensor
        dr = torch.tensor(discounted_rewards).float()    #.to(self.device)
        #print("dr ", dr.shape)
        # Get value as baseline
        V = self.critic(old_states).squeeze()
        # Compute advantage as total (discounted) return - value
        A = dr - V 
        # Rescale to unitary variance for a trajectory (axis=1)
        #A = (A - A.mean(axis=1).unsqueeze(1))/(A.std(axis=1).unsqueeze(1))
        #print("A ", A.shape)
        #print("log_probs ", log_probs.shape)
        # Compute - gradient
        policy_gradient = - log_probs*A
        #print("policy_gradient ", policy_gradient.shape)
        # Use it as loss
        policy_grad = torch.sum(policy_gradient)
        # barckprop and update
        self.actor_optim.zero_grad()
        policy_grad.backward()
        self.actor_optim.step()
        return
 
    
    
        
    