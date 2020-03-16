import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical

from networks import Actor, Critic, DiscreteActor, DiscreteCritic #custom module

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
        
        
    def get_action(self, state, return_log=False, greedy=True):
        probs = self.forward(state)
        if return_log:
            log_probs = torch.log(probs)
        probs = probs.detach().cpu().numpy().flatten()
        if greedy:
            # Choose action with higher prob
            action = np.argmax(probs) 
        else:
            # Sample action from discrete distribution
            action = np.random.choice(self.n_actions, p=probs)
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
        for log_prob, Gt in zip(log_probs, dr):
            policy_gradient.append(-log_prob*Gt) # "-" for minimization instead of maximization
        self.optim.zero_grad()
        policy_grad = torch.stack(policy_gradient).sum()
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
        
        
    def get_action(self, state, return_log=False, greedy=True):
        log_probs = self.forward(state)
        probs = torch.exp(log_probs)
        probs = probs.detach().numpy().flatten()
        if greedy:
            # Choose action with higher prob
            action = np.argmax(probs) 
        else:
            # Sample action from discrete distribution
            action = np.random.choice(self.n_actions, p=probs)
        if return_log:
            return action, log_probs.view(-1)[action]
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
        
        
    def get_action(self, state, return_log=False, greedy=True):
        dist = self.forward(state)
        if return_log:
            log_probs = torch.log(dist)
            
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
        
        
    def get_action(self, state, return_log=False, greedy=True):
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
    
class A2C_v0():
    """
    Implements Advantage Actor Critic RL agent. Updates to be executed step by step.
    
    Notes
    -----
    GPU implementation is just sketched; it works but it's slower than with CPU.
    """
    
    def __init__(self, observation_space, action_space, lr_actor, lr_critic, gamma, 
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
        
        self.n_actions = action_space
        self.discrete = discrete
        if self.discrete:
            self.actor = DiscreteActor(observation_space, action_space, project_dim)
            self.critic = DiscreteCritic(observation_space, project_dim)
        else:
            self.actor = Actor(observation_space, action_space)
            self.critic = Critic(observation_space)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.device = device 
        ### Not implemented ###
        #self.actor.to(self.device) 
        #self.critic.to(self.device)
        
    def get_action(self, state, return_log=False, greedy=False):
        if self.discrete:
            state = torch.from_numpy(state)
            log_probs = self.actor(state)
            dist = torch.exp(log_probs)
            probs = Categorical(dist)
            action =  probs.sample().item()
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)    #.to(self.device) 
            probs = self.actor(state)
            if return_log:
                log_probs = torch.log(probs)
            probs = probs.detach().cpu().numpy().flatten()
            if greedy:
                # Choose action with higher prob
                action = np.argmax(probs) 
            else:
                # Sample action from discrete distribution
                action = np.random.choice(self.n_actions, p=probs)
                
        if return_log:
            return action, log_probs.view(-1)[action]
        else:
            return action
    
    def update(self, reward, log_prob, state, new_state, done):
        # Wrap variables in tensors
        reward = torch.tensor(reward)
        if self.discrete:
            old_state = torch.tensor(state).unsqueeze(0)    
            new_state = torch.tensor(new_state).unsqueeze(0)
        else:
            old_state = torch.tensor(state).float().unsqueeze(0)    
            new_state = torch.tensor(new_state).float().unsqueeze(0)
        log_prob = torch.tensor([log_prob]) 
        # Update critic and then actor
        self.update_critic(reward, new_state, old_state, done)
        self.update_actor(reward, log_prob, new_state, old_state, done)
        return
    
    def update_critic(self, reward, new_state, old_state, done):
        # Predictions
        V_pred = self.critic(old_state).squeeze()
        # Targets
        V_trg = self.critic(new_state).squeeze()
        # done = 1 if new_state is a terminal state
        V_trg = (1-done)*self.gamma*V_trg + reward
        # MSE loss
        loss = (V_pred - V_trg).pow(2).sum()
        # backprop and update
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        return
    
    def update_actor(self, reward, log_prob, new_state, old_state, done):
        # compute advantage
        A = (1-done)*self.gamma*self.critic(new_state).squeeze() + reward - self.critic(old_state).squeeze()
        # compute gradient
        policy_gradient = - log_prob*A
        # backprop and update
        self.actor_optim.zero_grad()
        policy_gradient.backward()
        self.actor_optim.step()
        return
    
class A2C_v1():
    """
    Implements Advantage Actor Critic RL agent. Uses episode trajectories to update.
    
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
        self.actor = Actor(observation_space, action_space)
        self.critic = Critic(observation_space)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        
        self.device = device 
        ### Not implemented ###
        #self.actor.to(self.device) # move network to device
        #self.critic.to(self.device)
        
    def get_action(self, state, return_log=False, greedy=True):
        state = torch.from_numpy(state).float().unsqueeze(0)    #.to(self.device) 
        probs = self.actor(state)
        if return_log:
            log_probs = torch.log(probs)
        probs = probs.detach().cpu().numpy().flatten()
        if np.any(probs > 0.9):
            print("probs ", probs)
        if greedy:
            # Choose action with higher prob
            action = np.argmax(probs) 
        else:
            # Sample action from discrete distribution
            action = np.random.choice(self.n_actions, p=probs)
        
        if return_log:
            return action, log_probs.view(-1)[action]
        else:
            return action
    
    def update(self, rewards, log_probs, states, done):
        # Wrap variables in tensors
        old_states = torch.tensor(states[:,:-1]).float()    #.to(self.device)
        new_states = torch.tensor(states[:,1:]).float() 
        done = torch.LongTensor(done.astype(int))
        log_probs = torch.tensor(log_probs.astype(float)) #.to(self.device)
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
        
        # Predictions
        V_pred = self.critic(old_states).squeeze()
        # Targets
        V_trg = self.critic(new_states).squeeze()
        V_trg = (1-done)*self.gamma*V_trg + rewards
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
        # Get value as baseline
        V = self.critic(old_states).squeeze()
        # Compute advantage as total (discounted) return - value
        A = dr - V 
        # Rescale to unitary variance for a trajectory (axis=1)
        A = (A - A.mean(axis=1).unsqueeze(1))/(A.std(axis=1).unsqueeze(1))
        #print("A ", A)
        # Compute - gradient
        policy_gradient = - log_probs*A
        # Use it as loss
        policy_grad = torch.sum(policy_gradient)
        # barckprop and update
        self.actor_optim.zero_grad()
        policy_grad.backward()
        self.actor_optim.step()
        return
 
    
    
        
    