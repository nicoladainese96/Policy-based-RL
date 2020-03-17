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
 
    
    
class SeasideEnv:
    
    def __init__(self, x, y, initial, goal, R0):
        self.boundary = np.asarray([x, y])
        self.state = np.asarray(initial)
        self.goal = goal
        self.R0 = R0
        self.action_map = {
                            0: [0, 0],
                            1: [0, 1],
                            2: [0, -1],
                            3: [1, 0],
                            4: [-1, 0],
                            }
    # the agent makes an action (0 is stay, 1 is up, 2 is down, 3 is right, 4 is left)
    def step(self, action):
        #print("Current state:", self.state)
        if self.state[0] < 5:
            reward = self.R0
        else:
            reward = self.R0 *5
        movement = self.action_map[action]
        if (action == 0 and (self.state == self.goal).all()):
            reward = 1
        next_state = self.state + np.asarray(movement)
        if(self.check_boundaries(next_state)):
            reward = -1
        else:
            self.state = next_state
            
        #print("Current movement:", movement)
        #print("Reward obtained:", reward)
        return [self.state, reward]

    # map action index to movement
    def check_boundaries(self, state):
        out = len([num for num in state if num < 0])
        out += len([num for num in (self.boundary - np.asarray(state)) if num <= 0])
        return out > 0
    
class BridgeEnv:
    
    def __init__(self, x, y, initial, R0):
        self.boundary = np.asarray([x, y])
        self.state = np.asarray(initial)
        self.initial = np.asarray(initial)
        self.goal = [4,9]
        self.R0 = R0
        self.action_map = {
                            0: [0, 0],
                            1: [0, 1],
                            2: [0, -1],
                            3: [1, 0],
                            4: [-1, 0],
                            }
    # the agent makes an action (0 is stay, 1 is up, 2 is down, 3 is right, 4 is left)
    def step(self, action):
        #print("Current state:", self.state)
        reward = self.R0
        movement = self.action_map[action]
        if (action == 0 and (self.state == self.goal).all()):
            reward = 1
        next_state = self.state + np.asarray(movement)
        # boundary case
        if(self.check_boundaries(next_state)):
            reward = -1
        # cliff case -> re-start from beginning + big negative reward
        elif self.check_void(next_state):
            reward = -10
            self.state = self.initial
        else:
            self.state = next_state
            
        #print("Current movement:", movement)
        #print("Reward obtained:", reward)
        return [self.state, reward]

    # map action index to movement
    def check_boundaries(self, state):
        out = len([num for num in state if num < 0])
        out += len([num for num in (self.boundary - np.asarray(state)) if num <= 0])
        return out > 0
    
    def check_void(self, state):
        if state[0] in [i for i in range(1,9)] and state[1] in [i for i in range(2,8)]:
            return True
        else:
            return False    
    
    def get_optimal_action(self):
        print("\nCurrent state: ", self.state)
        optimal = np.zeros(self.n_actions)
        d0 = self.dist_to_goal(self.state)
        print("Original distance ", d0)
        # consider all actions
        for action in range(self.n_actions):
            # compute for each the resulting state
            print("\n\tAction : "+self.action_dict[action])
            movement = self.action_map[action]
            next_state = self.state + np.asarray(movement)
            print("\tNext state: ", next_state)
            if(self.check_boundaries(next_state)):
                print("\tAction allowed.")
                # if the state is admitted -> compute the distance to the goal 
                d = self.dist_to_goal(next_state)
                print("\tNew distance ", d)
                # if the new distance is smaller than the old one, is an optimal action (optimal = 1.)
                if d < d0:
                    optimal[action] = 1.
                else:
                    optimal[action] = 0.
            else:
                print("\tAction prohibited.")
                # oterwise is not (optimal = 0)
                optimal[action] = 0.
            
            print("\tAction is optimal: ", optimal[action] == 1)
        print("\noptimal ", optimal)
        # once we have the vector of optimal, divide them by the sum
        probs = optimal/optimal.sum()
        print("probs ", probs)
        # finally sample the action and return it together with the log of the probability
        opt_action = np.random.choice(self.n_actions, p=probs)
        print("Action chosen: ", opt_action, "("+self.action_dict[opt_action]+")\n")
        return opt_action