import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical

from networks import Actor, Critic #custom module

class A2C():
    """
    Advantage Actor-Critic RL agent. 
    
    Notes
    -----
    * GPU implementation is still work in progress.
    * Always uses 2 separate networks for the critic,one that learns from new experience 
      (student/critic) and the other one (critic_target/teacher)that is more conservative 
      and whose weights are updated through an exponential moving average of the weights 
      of the critic, i.e.
          target.params = (1-tau)*target.params + tau* critic.params
    * In the case of Monte Carlo estimation the critic_target is never used
    * Possible to use twin networks for the critic and the critic target for improved 
      stability. Critic target is used for updates of both the actor and the critic and
      its output is the minimum between the predictions of its two internal networks.
      
    """ 
    
    def __init__(self, observation_space, action_space, lr, gamma, TD=True, discrete=False, 
                 project_dim=8, hiddens = [64,32], twin=False, tau = 1., device='cpu', debug=False):
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
        TD: bool (default=True)
            If True, uses Temporal Difference for the critic's estimates
            Otherwise uses Monte Carlo estimation
        discrete: bool (default=False)
            If True, adds an embedding layer both in the actor 
            and the critic networks before processing the state.
            Should be used if the state is a simple integer in [0, observation_space -1]
        project_dim: int (default=8)
            Number of dimensions of the embedding space (e.g. number of dimensions of
            embedding(state) ). Higher dimensions are more expressive.
        hiddens: list of int (default = [64,32])
            List containing the number of neurons of each linear hidden layer.
            Same architecture is considered for the actor and the critic, except from the 
            output layer, than in one case has the dimension of the action space and a LogSoftmax
            activation, in the other outputs a scalar (state value)
        twin: bool (default=False)
            Enables twin networks both for critic and critic_target
        tau: float in [0,1] (default = 1.)
            Regulates how fast the critic_target gets updates, i.e. what percentage of the weights
            inherits from the critic. If tau=1., critic and critic_target are identical 
            at every step, if tau=0. critic_target is unchangable. 
            As a default this feature is disabled setting tau = 1, but if one wants to use it a good
            empirical value is 0.005.
        device: str in {'cpu','cuda'}
            Implemented, but GPU slower than CPU because it's difficult to optimize a RL agent without
            a replay buffer, that can be used only in off-policy algorithms.
        """
        
        self.gamma = gamma
        self.lr = lr
        
        self.n_actions = action_space
        self.discrete = discrete
        self.TD = TD
        self.twin = twin 
        self.tau = tau
        
        self.actor = Actor(observation_space, action_space, discrete, project_dim, hiddens=hiddens)
        self.critic = Critic(observation_space, discrete, project_dim, twin, hiddens=hiddens)
        
        if self.TD:
            self.critic_trg = Critic(observation_space, discrete, project_dim, twin, target=True, hiddens=hiddens)

            # Init critic target identical to critic
            for trg_params, params in zip(self.critic_trg.parameters(), self.critic.parameters()):
                trg_params.data.copy_(params.data)
            
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        self.device = device 
        self.actor.to(self.device) 
        self.critic.to(self.device)
        if self.TD:
            self.critic_trg.to(self.device)
        
        if debug:
            print("="*10 +" A2C HyperParameters "+"="*10)
            print("Discount factor: ", self.gamma)
            print("Learning rate: ", self.lr)
            print("Action space: ", self.n_actions)
            print("Discrete state space: ", self.discrete)
            print("Temporal Difference learning: ", self.TD)
            print("Twin networks: ", self.twin)
            print("Update critic target factor: ", self.tau)
            print("Device used: ", self.device)
            print("\n\n"+"="*10 +" A2C Architecture "+"="*10)
            print("Actor architecture: \n", self.actor)
            print("Critic architecture: \n",self.critic)
            print("Critic target architecture: ")
            if self.TD:
                print(self.critic_trg)
            else:
                print("Not used")
        
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
            state = torch.from_numpy(state).to(self.device)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
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
            old_states = torch.tensor(states[:-1]).to(self.device)
            new_states = torch.tensor(states[1:]).to(self.device)
        else:
            old_states = torch.tensor(states[:,:-1]).float().to(self.device)
            new_states = torch.tensor(states[:,1:]).float().to(self.device)
            
        done = torch.LongTensor(done.astype(int)).to(self.device)
        log_probs = torch.stack(log_probs).to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        
        ### Update critic and then actor ###
        critic_loss = self.update_critic_TD(rewards, new_states, old_states, done)
        actor_loss = self.update_actor_TD(rewards, log_probs, new_states, old_states, done)
        
        return critic_loss, actor_loss
    
    def update_critic_TD(self, rewards, new_states, old_states, done):
        
        # Compute loss 
        
        with torch.no_grad():
            V_trg = self.critic_trg(new_states).squeeze()
            V_trg = (1-done)*self.gamma*V_trg + rewards
            V_trg = V_trg.squeeze()
            
        if self.twin:
            #print(self.critic(old_states))
            V1, V2 = self.critic(old_states)
            loss1 = 0.5*F.mse_loss(V1.squeeze(), V_trg)
            loss2 = 0.5*F.mse_loss(V2.squeeze(), V_trg)
            loss = loss1 + loss2
        else:
            V = self.critic(old_states).squeeze()
            loss = F.mse_loss(V, V_trg)
        
        # Backpropagate and update
        
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        
        # Update critic_target: (1-tau)*old + tau*new
        
        for trg_params, params in zip(self.critic_trg.parameters(), self.critic.parameters()):
                trg_params.data.copy_((1.-self.tau)*trg_params.data + self.tau*params.data)
        
        return loss.item()
    
    def update_actor_TD(self, rewards, log_probs, new_states, old_states, done):
        
        # Compute gradient 
        
        if self.twin:
            V1, V2 = self.critic(old_states)
            V_pred = torch.min(V1.squeeze(), V2.squeeze())
            V1_new, V2_new = self.critic(new_states)
            V_new = torch.min(V1_new.squeeze(), V2_new.squeeze())
            V_trg = (1-done)*self.gamma*V_new + rewards
        else:
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
        
        dr = torch.tensor(discounted_rewards).float().to(self.device) 
        
        if self.discrete:
            old_states = torch.tensor(states[:-1]).to(self.device)
            new_states = torch.tensor(states[1:]).to(self.device)
        else:
            old_states = torch.tensor(states[:,:-1]).float().to(self.device)
            new_states = torch.tensor(states[:,1:]).float().to(self.device)
            
        done = torch.LongTensor(done.astype(int)).to(self.device)
        log_probs = torch.stack(log_probs).to(self.device)
        
        ### Update critic and then actor ###
        
        critic_loss = self.update_critic_MC(dr, old_states)
        actor_loss = self.update_actor_MC(dr, log_probs, old_states)
        
        return critic_loss, actor_loss
    
    def update_critic_MC(self, dr, old_states):

        # Compute loss
        
        if self.twin:
            V1, V2 = self.critic(old_states)
            V_pred = torch.min(V1.squeeze(), V2.squeeze())
        else:
            V_pred = self.critic(old_states).squeeze()
            
        loss = F.mse_loss(V_pred, dr)
        
        # Backpropagate and update
        
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        
        return loss.item()
    
    def update_actor_MC(self, dr, log_probs, old_states):
        
        # Compute gradient 
        
        if self.twin:
            V1, V2 = self.critic(old_states)
            V_pred = torch.min(V1.squeeze(), V2.squeeze())
        else:
            V_pred = self.critic(old_states).squeeze()
            
        A = dr - V_pred
        policy_gradient = - log_probs*A
        policy_grad = torch.sum(policy_gradient)
 
        # Backpropagate and update
    
        self.actor_optim.zero_grad()
        policy_grad.backward()
        self.actor_optim.step()
        
        return policy_grad.item()

    