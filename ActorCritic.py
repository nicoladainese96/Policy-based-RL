import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical

from networks import Actor, Critic, DiscreteActor, DiscreteCritic #custom module

class A2C():
    """
    Implements Advantage Actor Critic RL agent. 
    
    Notes
    -----
    GPU implementation is just sketched; it works but it's slower than with CPU.
    """
    
    def __init__(self, observation_space, action_space, lr, gamma, TD=True,
                 device='cpu', discrete=False, project_dim=8):
        """
        Parameters
        ----------
        observation_space: int
            Number of flattened entries of the state
        action_space: int
            Number of (discrete) possible actions to take
        lr: float in [0,1]
            Learning rate
        gamma: float in [0,1]
            Discount factor
        TD: bool
            If True, uses Temporal Difference for the critic's estimates
            Otherwise uses Monte Carlo estimation
        device: str in {'cpu','cuda'}
            Not implemented at the moment
        discrete: bool
            If True, adds an embedding layer both in the actor 
            and the critic networks before processing the state.
            Should be used if the state is a simple integer in [0, observation_space -1]
        project_dim: int
            Number of dimensions of the embedding space (e.g. number of dimensions of
            embedding(state) ). Higher dimensions are more expressive.
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
        
        self.TD = TD
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
        """
        Makes a tensor out of a numpy array state and then forward
        it with the actor network.
        
        Parameters
        ----------
        state:
            If self.discrete is True state.shape = (episode_len,)
            Otherwise state.shape = (episode_len, observation_space)
        """
        if self.discrete:
            state = torch.from_numpy(state)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0) 
        log_probs = self.actor(state)
        return log_probs
    
    def update(self, *args):
        if self.TD:
            critic_loss, actor_loss = self.update_TD(*args)
        else:
            critic_loss, actor_loss = self.update_MC(*args)
        
        return critic_loss, actor_loss
    
    def update_TD(self, rewards, log_probs, states, done):   
        
        ### Wrap variables into tensors ###
        
        if self.discrete:
            old_states = torch.tensor(states[:-1])
            new_states = torch.tensor(states[1:])
        else:
            old_states = torch.tensor(states[:,:-1]).float()
            new_states = torch.tensor(states[:,1:]).float()
        done = torch.LongTensor(done.astype(int))
        log_probs = torch.stack(log_probs)
        rewards = torch.tensor(rewards) 
        
        ### Update critic and then actor ###
        
        critic_loss = self.update_critic_TD(rewards, new_states, old_states, done)
        actor_loss = self.update_actor_TD(rewards, log_probs, new_states, old_states, done)
        
        return critic_loss, actor_loss
    
    def update_critic_TD(self, rewards, new_states, old_states, done):
        
        # Compute loss 
        
        V_pred = self.critic(old_states).squeeze()
        V_trg = self.critic(new_states).squeeze()
        V_trg = (1-done)*self.gamma*V_trg + rewards
        loss = torch.sum((V_pred - V_trg)**2)
        
        # Backpropagate and update
        
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        
        return loss.item()
    
    def update_actor_TD(self, rewards, log_probs, new_states, old_states, done):
        
        # Compute gradient 
        
        V_pred = self.critic(old_states).squeeze()
        V_trg = (1-done)*self.gamma*self.critic(new_states).squeeze()  + rewards
        A = V_trg - V_pred
        policy_gradient = - log_probs*A
        policy_grad = torch.sum(policy_gradient)
 
        # Backpropagate and update
    
        self.actor_optim.zero_grad()
        policy_grad.backward()
        self.actor_optim.step()
        
        return policy_grad.item()
    
    def update_MC(self, rewards, log_probs, states, done):   
        
        ### Compute MC discounted returns ###
        
        Gamma = np.array([self.gamma**i for i in range(rewards.shape[0])])
        # reverse everything to use cumsum in right order, then reverse again
        Gt = np.cumsum(rewards[::-1]*Gamma[::-1])[::-1]
        # Rescale so that present reward is never discounted
        discounted_rewards =  Gt/Gamma
        
        ### Wrap variables into tensors ###
        dr = torch.tensor(discounted_rewards).float() 
        
        if self.discrete:
            old_states = torch.tensor(states[:-1])
            new_states = torch.tensor(states[1:])
        else:
            old_states = torch.tensor(states[:,:-1]).float()
            new_states = torch.tensor(states[:,1:]).float()
        done = torch.LongTensor(done.astype(int))
        log_probs = torch.stack(log_probs)
        
        ### Update critic and then actor ###
        
        critic_loss = self.update_critic_MC(dr, old_states)
        actor_loss = self.update_actor_MC(dr, log_probs, old_states)
        
        return critic_loss, actor_loss
    
    def update_critic_MC(self, dr, old_states):

        # Compute loss
        
        V_pred = self.critic(old_states).squeeze()
        loss = torch.sum((V_pred - dr)**2)
        
        # Backpropagate and update
        
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        
        return loss.item()
    
    def update_actor_MC(self, dr, log_probs, old_states):
        
        # Compute gradient 
        
        V = self.critic(old_states).squeeze()
        A = dr - V 
        policy_gradient = - log_probs*A
        policy_grad = torch.sum(policy_gradient)
 
        # Backpropagate and update
    
        self.actor_optim.zero_grad()
        policy_grad.backward()
        self.actor_optim.step()
        
        return policy_grad.item()

    