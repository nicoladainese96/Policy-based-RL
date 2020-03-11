import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 

from networks import Actor, Critic #custom module

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
        #print("state (before torching) ", state.shape)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device) 
        #print("state (after torching) ", state.shape)
        return self.net(state) #.to('cpu')       
    
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
            policy_gradient.append(-log_prob*Gt) # - for minimization instead of maximization
        
        #print("policy_gradient (before torching): ", np.array(policy_gradient).shape) 
        #print("policy_gradient (after torching): ", torch.stack(policy_gradient).shape) 
        self.optim.zero_grad()
        policy_grad = torch.stack(policy_gradient).sum()
        policy_grad.backward()
        self.optim.step()


class A2C_v0():
    """
    Implements Advantage Actor Critic RL agent.
    
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
        #self.actor.to(self.device) # move network to device
        #self.critic.to(self.device)
        
    def get_action(self, state, return_log=False, greedy=False):
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
        reward = torch.tensor(reward)
        old_state = torch.tensor(state).float().unsqueeze(0)    #try to remove .float()
        new_state = torch.tensor(new_state).float().unsqueeze(0)
        #done = torch.LongTensor([done)
        log_prob = torch.tensor([log_prob]) #.to(self.device)
        #print("reward ", reward)
        #print("old_states ", old_state.shape)
        #print("new_states ", new_state.shape)
        #print("done ", done)
        #print("log_prob ", log_prob)
        self.update_critic(reward, new_state, old_state, done)
        self.update_actor(reward, log_prob, new_state, old_state, done)
        return
    
    def update_critic(self, reward, new_state, old_state, done):
        """
        Minimize \sum_{t=0}^{T-1}(rewards[t] + gamma V(new_states[t]) - V(old_states[t]) )**2
        where V(state) is the prediction of the critic.
        
        Parameters
        ----------
        reward: shape (T,)
        old_states, new_states: shape (T, observation_space)
        """
        # Predictions
        V_pred = self.critic(old_state).squeeze()
        #print("V_pred \n", V_pred)
        # Targets
        V_trg = self.critic(new_state).squeeze()
        #print("V_trg (init) ", V_trg)
        # done = 1 if new_state is a terminal state
        V_trg = (1-done)*self.gamma*V_trg + reward
        #print("V_trg (final) ", V_trg)
        #print("rewards ", rewards.shape)
        #V_trg = V_trg + rewards
        #print("V_trg (after +r) \n", V_trg)
        
        # MSE loss
        loss = (V_pred - V_trg).pow(2).sum()
        #loss = torch.sum((V_pred - dr)**2, axis = 1)
        #print("loss ", loss)
        #loss = torch.mean(loss)
        #print("loss \n", loss)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        return
    
    def update_actor(self, reward, log_prob, new_state, old_state, done):
        
        A = (1-done)*self.gamma*self.critic(new_state).squeeze() + reward - self.critic(old_state).squeeze()
        #print("A \n", A)
        #A = A/A.std(axis=1).unsqueeze(1)
        #print("A ", A)
        policy_gradient = - log_prob*A
        #print("policy_gradient ", policy_gradient)
        #policy_grad = policy_gradient.item()
        #print("policy_gradient ", policy_grad)
        
        self.actor_optim.zero_grad()
        policy_gradient.backward()
        self.actor_optim.step()
        return
    
class A2C_v1():
    """
    Implements Advantage Actor Critic RL agent.
    
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
        #self.actor.to(self.device) # move network to device
        #self.critic.to(self.device)
        
    def get_action(self, state, return_log=False, greedy=True):
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
    
    def update(self, rewards, log_probs, states, done):
        old_states = torch.tensor(states[:,:-1]).float()    #.to(self.device)
        new_states = torch.tensor(states[:,1:]).float() 
        done = torch.LongTensor(done.astype(int))
        #print("done ", done)
        #print("old_states ", old_states)
        #print("new_states ", new_states)
        log_probs = torch.tensor(log_probs.astype(float)) #.to(self.device)
        self.update_actor(rewards, log_probs, new_states, old_states)
        self.update_critic(rewards, new_states, old_states, done)
        
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
        #print("V_pred \n", V_pred)
        # Targets
        V_trg = self.critic(new_states).squeeze()
        #print("V_trg (init) ", V_trg)
        V_trg = (1-done)*self.gamma*V_trg + rewards
        #print("V_trg (final) ", V_trg)
        #print("rewards ", rewards.shape)
        #V_trg = V_trg + rewards
        #print("V_trg (after +r) \n", V_trg)
        
        # MSE loss
        loss = torch.sum((V_pred - V_trg)**2)
        #loss = torch.sum((V_pred - dr)**2, axis = 1)
        #print("loss ", loss)
        #loss = torch.mean(loss)
        #print("loss \n", loss)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        return
    
    def update_actor(self, rewards, log_probs, new_states, old_states):
        
        Gamma = np.array([self.gamma**i for i in range(rewards.shape[1])]).reshape(1,-1)
        #print("Gamma ", Gamma)
        Gt = np.cumsum(rewards[:,::-1]*Gamma[:,::-1], axis=1)[:,::-1]
        #print("rewards ", rewards)
        #print("Gt ", Gt)
        discounted_rewards =  Gt/Gamma
        dr = torch.tensor(discounted_rewards).float()    #.to(self.device)
        #print("dr ", dr)
        V = self.critic(old_states).squeeze()
        #print("V (actor) \n", V)
        A = dr - V # Different way of computing the advantage
        #print("A \n", A)
        A = A/A.std(axis=1).unsqueeze(1)
        #print("A ", A)
        policy_gradient = - log_probs*A
        #print("policy_gradient ", policy_gradient)
        policy_grad = torch.sum(policy_gradient)
        
        self.actor_optim.zero_grad()
        policy_grad.backward()
        self.actor_optim.step()
        return
 
    
    
        
    