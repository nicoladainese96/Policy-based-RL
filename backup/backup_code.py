import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 

from networks import Actor, Critic #custom module

def train_cartpole_A2C_v0(n_epochs = 100, lr_actor = 0.01, lr_critic = 0.01, gamma = 0.99):
    # Create environment
    env = gym.make("CartPole-v1")
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    # Init agent
    agent = agents.A2C_v0(observation_space, action_space, lr_actor, lr_critic, gamma)
    performance = []
    for e in range(n_epochs):
        # Reset environment (start of an episode)
        state = env.reset()
        rewards = []

        steps = 0
        while True:
            action, log_prob = agent.get_action(state, return_log = True)
            new_state, reward, terminal, info = env.step(action) # gym standard step's output

            #if terminal and 'TimeLimit.truncated' not in info:
            #    reward = -1

            rewards.append(reward)
            agent.update(reward, log_prob, state, new_state, terminal)
            
            if terminal:
                break

            state = new_state
            
            
        rewards = np.array(rewards)
        performance.append(np.sum(rewards))
        if (e+1)%10 == 0:
            print("Episode %d - reward: %.0f"%(e+1, np.mean(performance[-10:])))

    return agent, np.array(performance)

class A2C_v0(): # DOES NOT WORK STILL
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
         
    def update(self, reward, log_prob, state, new_state, done):
        # Wrap variables in tensors
        reward = torch.tensor(reward)
        if self.discrete:
            old_state = torch.tensor(state).unsqueeze(0)    
            new_state = torch.tensor(new_state).unsqueeze(0)
        else:
            old_state = torch.tensor(state).float().unsqueeze(0)    
            new_state = torch.tensor(new_state).float().unsqueeze(0)
        #log_prob = torch.tensor([log_prob]) # THIS DETACHES THE TENSOR!!
        log_prob = log_prob.view(1,1)
        # Update critic and then actor
        self.update_critic(reward, new_state, old_state, done)
        self.update_actor(reward, log_prob, new_state, old_state, done)
        return
    
    def update_critic(self, reward, new_state, old_state, done):
        # Predictions
        V_pred = self.critic(old_state).squeeze()
        #print("V_pred ", V_pred)
        # Targets
        V_trg = self.critic(new_state).squeeze()
        #print("V_trg (net) ", V_trg)
        # done = 1 if new_state is a terminal state
        V_trg = (1-done)*self.gamma*V_trg + reward
        V_trg = V_trg.detach()
        #print("V_trg (+r) ", V_trg)
        # MSE loss
        loss = (V_pred - V_trg).pow(2).sum()
        #print("loss ", loss)
        # backprop and update
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        return
    
    def update_actor(self, reward, log_prob, new_state, old_state, done):
        # compute advantage
        A = (1-done)*self.gamma*self.critic(new_state).squeeze() + reward - self.critic(old_state).squeeze()
        #print("Advantage ", A)
        # compute gradient
        policy_gradient = - log_prob*A
        #print("policy_gradient ", policy_gradient)
        # backprop and update
        self.actor_optim.zero_grad()
        policy_gradient.backward()
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
    
# entropy part

def play_episode_ent(agent, env, return_states=False, greedy=True):
    # Reset environment (start of an episode)
    state = env.reset()
    rewards = []
    log_probs = []
    distributions = []
    done = []
    
    if return_states:
        states = [state]
        
    steps = 0
    while True:
        action, log_prob, prob_distr = agent.get_action(state, return_log = True, greedy=greedy)
        new_state, reward, terminal, info = env.step(action) # gym standard step's output
        
        if return_states:
            states.append(new_state)
            
        #if terminal and 'TimeLimit.truncated' not in info:
        #    reward = -1
            
        rewards.append(reward)
        log_probs.append(log_prob)
        distributions.append(prob_distr)
        done.append(terminal)
        
        if terminal:
            break
            
        state = new_state
       
    rewards = np.array(rewards)
    log_probs = np.array(log_probs)
    #distributions = np.array(distributions)
    done = np.array(done)
    
    if return_states:
        return rewards, log_probs, np.array(states), distributions, done
    else:
        return rewards, log_probs, distributions, done
    
def train_cartpole_entropy(n_episodes = 100, lr = 0.01, gamma = 0.99, H=1e-3, greedy=False):
    # Create environment
    env = gym.make("CartPole-v1")
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    # Init agent
    agent = agents.PolicyGradEnt(observation_space, action_space, lr, gamma, H)
    performance = []
    losses = []
    steps_log = []
    for e in range(n_episodes):
        rewards, log_probs, distributions, _ = play_episode_ent(agent, env, greedy=greedy)
        steps_log.append(len(rewards))
        performance.append(np.sum(rewards))
        if (e+1)%10 == 0:
            print("Episode %d - reward: %.0f"%(e+1, np.mean(performance[-10:])))
        
        loss = agent.update(rewards, log_probs, distributions)
        losses.append(loss)
    return agent, np.array(performance), np.array(losses), np.array(steps_log)

%%time
trained_agentPG, cumulative_rewardPG, lossesPG, steps_log = train_cartpole_entropy(n_episodes = 500, lr=5e-3, H=5e-3)


# Actor and critic

class Actor(nn.Module):
    """
    Network used to parametrize the policy of an agent.
    Uses 3 linear layers, the first 2 with ReLU activation,
    the third with softmax.
    """
    
    def __init__(self, observation_space, action_space):
        """
        Parameters
        ----------
        observation_space: int
            Number of flattened entries of the state
        action_space: int
            Number of (discrete) possible actions to take
        """
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(observation_space, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, action_space)
    
    def forward(self, state):
        out = F.relu(self.linear1(state))
        out = F.relu(self.linear2(out))
        log_probs = F.log_softmax(self.linear3(out), dim=1)
        return log_probs

class DiscreteActor(nn.Module):
    """
    Network used to parametrize the policy of an agent.
    Uses 3 linear layers, the first 2 with ReLU activation,
    the third with softmax.
    """
    
    def __init__(self, observation_space, action_space, project_dim=4):
        """
        Parameters
        ----------
        observation_space: int
            Number of flattened entries of the state
        action_space: int
            Number of (discrete) possible actions to take
        """
        super(DiscreteActor, self).__init__()
        self.embedding = nn.Embedding(observation_space, project_dim)
        self.linear1 = nn.Linear(project_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, action_space)
    
    def forward(self, state):
        state = self.embedding(state)
        out = F.relu(self.linear1(state))
        out = F.relu(self.linear2(out))
        log_probs = F.log_softmax(self.linear3(out), dim=1)
        return log_probs
    
    
class Critic(nn.Module):
    """
    Network used to parametrize the Critic of an Actor-Critic agent.
    Uses 3 linear layers, only the first 2 with ReLU activation.
    Returns the value of a state.
    """
    
    def __init__(self, observation_space):
        """
        Parameters
        ----------
        observation_space: int
            Number of flattened entries of the state
        hidden_dim: int
            Size of the hidden layer
        """
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(observation_space, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 1)
    
    def forward(self, state):
        out = F.relu(self.linear1(state))
        out = F.relu(self.linear2(out))
        return self.linear3(out) 
    
class DiscreteCritic(nn.Module):
    """
    Network used to parametrize the Critic of an Actor-Critic agent.
    Uses 3 linear layers, only the first 2 with ReLU activation.
    Returns the value of a state.
    """
    
    def __init__(self, observation_space, project_dim):
        """
        Parameters
        ----------
        observation_space: int
            Number of flattened entries of the state
        hidden_dim: int
            Size of the hidden layer
        """
        super(DiscreteCritic, self).__init__()
        self.embedding = nn.Embedding(observation_space, project_dim)
        self.linear1 = nn.Linear(project_dim, 8)
        self.linear2 = nn.Linear(8, 8)
        self.linear3 = nn.Linear(8, 1)
    
    def forward(self, state):
        out = self.embedding(state)
        out = F.relu(self.linear1(out))
        out = F.relu(self.linear2(out))
        return self.linear3(out) 
    
    
class Critic(nn.Module):
    """
    Network used to parametrize the Critic of an Actor-Critic agent.
    Uses 3 linear layers, only the first 2 with ReLU activation.
    Returns the value of a state.
    """
    
    def __init__(self, observation_space):
        """
        Parameters
        ----------
        observation_space: int
            Number of flattened entries of the state
        hidden_dim: int
            Size of the hidden layer
        """
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(observation_space, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 1)
    
    def forward(self, state):
        out = F.relu(self.linear1(state))
        out = F.relu(self.linear2(out))
        return self.linear3(out) 
    
class DiscreteCritic(nn.Module):
    """
    Network used to parametrize the Critic of an Actor-Critic agent.
    Uses 3 linear layers, only the first 2 with ReLU activation.
    Returns the value of a state.
    """
    
    def __init__(self, observation_space, project_dim):
        """
        Parameters
        ----------
        observation_space: int
            Number of flattened entries of the state
        hidden_dim: int
            Size of the hidden layer
        """
        super(DiscreteCritic, self).__init__()
        self.embedding = nn.Embedding(observation_space, project_dim)
        self.linear1 = nn.Linear(project_dim, 8)
        self.linear2 = nn.Linear(8, 8)
        self.linear3 = nn.Linear(8, 1)
    
    def forward(self, state):
        out = self.embedding(state)
        out = F.relu(self.linear1(out))
        out = F.relu(self.linear2(out))
        return self.linear3(out) 
        