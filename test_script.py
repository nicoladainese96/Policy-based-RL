import HP_tuning as hp
import numpy as np
import torch
from importlib import reload

# spans different magnitude scales, use logaritmic distribution for invariance of scale
lr_P_dict = dict(discrete=False,
                var_type='float',
                distribution='log',
                x_min = 1e-4,
                x_max = 1e-2)

# spans different magnitude scales, use logaritmic distribution for invariance of scale
tau_P_dict = dict(discrete=False,
                var_type='float',
                distribution='log',
                x_min = 1e-3,
                x_max = 1e-0)

possible_architectures = np.array([
                            [64,32,16],
                            [16,8,4],
                            [64,32],
                            [64,16],
                            [64,8],
                            [32,16],
                            [32,8],
                            [16,8],
                            [16,4],
                            [8,4]   
                        ])

probs = np.ones(10)/10

architecture_P_dict = dict(discrete=True,
                           var_type='str',
                           distribution='custom',
                           elements=possible_architectures,
                           p = probs)

P_dict = dict(lr=lr_P_dict, 
              tau=tau_P_dict,
              hiddens=architecture_P_dict)

n_samples = 2

params = {}
for key in P_dict.keys():
    P = hp.prior_distr(**P_dict[key])
    params[key] = P.sample(n_samples)

# now we make a list of dictionary instead of a dictionary of lists and we also add the last constant parameters
list_of_dict = []
for i in range(n_samples):
    d = {'TD' : True, 'twin' : True, 'gamma' : 0.99}
    for key in params.keys():
        d[key] = params[key][i]
    list_of_dict.append(d)

print("Number of combinations: ", len(list_of_dict))

def print_parameters(params):
    print("Parameters: ")
    print('='*75)
    for key in params:
        if (key == 'hiddens'):
            print(key, params[key])
        else:
            print(key, '\t', params[key])
    print('='*75)

def print_HP_score(params,score,dev):
    print_parameters(params)
    print("Score: %.4f +/- %.4f"%(score,dev))

flag = True #set to True to see all combinations
if flag == True:
    for params in list_of_dict:
        print()
        print_parameters(params)

import numpy as np
import torch
import gym
import ActorCritic
import time

def play_episode(agent, env, return_states=False):
    # Reset environment (start of an episode)
    state = env.reset()
    rewards = []
    log_probs = []
    done = []
    
    if return_states:
        states = [state]
        
        
    steps = 0
    while True:
        steps += 1
        action, log_prob = agent.get_action(state, return_log = True)
        new_state, reward, terminal, info = env.step(action) # gym standard step's output
        
        if return_states:
            states.append(new_state)
            
        if terminal and 'TimeLimit.truncated' not in info:
            # give -1 if cartpole falls but not if episode is truncated
            reward = -1 
            
        rewards.append(reward)
        log_probs.append(log_prob)
        done.append(terminal)
        
        if terminal or steps > 500:
            break
            
        state = new_state
        
    rewards = np.array(rewards)
    done = np.array(done)
    
    if return_states:
        return rewards, log_probs, np.array(states), done
    else:
        return rewards, log_probs, done

def train_cartpole_A2C(agent, env, n_episodes = 1000):
    performance = []
    print("n_episodes: ", n_episodes)
    for e in range(n_episodes):
        print("Episode %d started"%(e+1))
        rewards, log_probs, states, done = play_episode(agent, env, return_states=True)
        print("Episode %d finished"%(e+1))
        performance.append(np.sum(rewards))
        print("len(performance) ", len(performance))
        if (e+1)%100 == 0:
            print("Episode %d - reward: %.0f"%(e+1, np.mean(performance[-100:])))
        if e>n_episodes-2:
            print(rewards)
            print(log_probs)
            print(np.array([states]))
            print(done)
        agent.update(rewards, log_probs, np.array([states]) , done)
        print("Update done.")
        
    print("check")
    performance = np.array(performance)
    L = n_episodes // 6
    return performance, performance[-L:].mean(), performance[-L:].std()/np.sqrt(L)

def evaluate_agent(n_runs, n_episodes, **HPs):
    runs_scores = []
    asymptotic_score = []
    asymptotic_std = []
    
    for r in range(n_runs):
        
        print("\nRun %d/%d: "%(r+1, n_runs)) 
        
        # Create environment
    
        env = gym.make("CartPole-v1")
        observation_space = env.observation_space.shape[0]
        action_space = env.action_space.n
    
        # Init agent
    
        agent = ActorCritic.A2C(observation_space, action_space, **HPs)
    
        # Evaluate agent on a single run 
        
        perf, asymptotic_perf, asymptotic_err = train_cartpole_A2C(agent, env, n_episodes)
        print("perf ", perf)
        print("asymptotic_perf ", asymptotic_perf)
        print("asymptotic_err ", asymptotic_err)
        print("Train finished")
        runs_scores.append(perf)
        asymptotic_score.append(asymptotic_perf)
        asymptotic_std.append(asymptotic_std)
        
    asymptotic_score = np.array(asymptotic_score).mean()
    asymptotic_std = np.array(asymptotic_std).mean()
    return np.array(runs_scores), asymptotic_score, asymptotic_std

def print_parameters(params):
    print("Parameters: ")
    print('='*75)
    for key in params:
        if (key == 'hiddens'):
            print(key, params[key])
        else:
            print(key, '\t', params[key])
    print('='*75)

def print_HP_score(params,score,dev):
    print_parameters(params)
    print("Score: %.4f +/- %.4f"%(score,dev))

def HP_Search(n_runs, n_episodes, list_of_HP_dict):
    
    HP_scores = []
    HP_asymptotic_score = []
    HP_asymptotic_std = []
    
    for i, HP in enumerate(list_of_HP_dict):
        
        runs_scores, asymptotic_score, asymptotic_std = evaluate_agent(n_runs, n_episodes, **HP)
        print("agent evaluated")
        print_HP_score(HP, asymptotic_score, asymptotic_std)
        HP_scores.append(runs_scores)
        HP_asymptotic_score.append(asymptotic_score)
        HP_asymptotic_std.append(asymptotic_std)
    
    return HP_scores, HP_asymptotic_score, HP_asymptotic_std

class prior_distr():
    def __init__(self, discrete, var_type, distribution, **kwargs):
        self.discrete = discrete
        self.type = var_type
        if discrete:
            if var_type == 'int':
                if distribution == 'exp':
                    self.distr = distribution
                    self.N_min = kwargs['N_min']
                    self.N_max = kwargs['N_max']
                    self.alpha = kwargs['alpha']
                    self.elements = np.arange(self.N_min,self.N_max+1)
                elif distribution == 'uniform':
                    self.distr = distribution
                    self.N_min = kwargs['N_min']
                    self.N_max = kwargs['N_max']
                    self.elements = np.arange(self.N_min,self.N_max+1)
                    self.probabilities = np.full(len(self.elements), 1/len(self.elements))
                elif distribution == 'custom':
                    self.distr = distribution
                    self.elements = kwargs['elements']
                    if (np.abs(kwargs['p'].sum()-1) < 1e-4):
                        self.probabilities = kwargs['p']
                    else:
                        raise Exception ('Total probability must be equal to 1.')
                else:
                    raise Exception('Variable \'distribution\' must be \'exp\', \'uniform\' or \'custom\'.')
            elif var_type == 'str':
                self.elements = kwargs['elements']
                self.distr = distribution
                if distribution == 'uniform':
                    self.probabilities = np.full(len(self.elements), 1/len(self.elements))
                elif distribution == 'custom':
                    if (np.abs(kwargs['p'].sum()-1) < 1e-4):
                        self.probabilities = kwargs['p']
                    else:
                        raise Exception ('Total probability must be equal to 1.')
                else:
                    raise Exception('Variable \'distribution\' must be \'uniform\' or \'custom\'.')
            else:
                raise Exception('Type must be \'int\' or \'str\'.')
        else:
            # continuous case
            if var_type == 'float':
                if distribution == 'uniform':
                    self.distr = distribution
                    self.x_min = kwargs['x_min']
                    self.x_max = kwargs['x_max']
                elif distribution == 'log':
                    self.distr = distribution
                    self.x_min = kwargs['x_min']
                    self.x_max = kwargs['x_max']
                else:
                    raise Exception('Distribution must be \'uniform\' or \'log\'.')
            else:
                raise Exception('Continuous type must be \'float\'.')
        
    def sample(self, n_samples):
        if self.discrete == True:
            def sample_from_discrete_distr(n_samples, p_cum):
                u = np.random.rand(n_samples)
                mask = np.tile(p_cum[:,np.newaxis], (1,len(u))) > u
                samples = self.elements[np.argmax(mask, axis=0)]
                return samples

            if (self.type == 'int') and (self.distr == 'exp'): # only case in which self.probabilities is not defined
                def exp_distr(n):
                    exp_of_Ns = np.exp(-self.alpha*self.elements)
                    norm_factor = exp_of_Ns.sum() # 1/(e^alpha - 1)
                    p_of_n = np.exp(-self.alpha*n)/norm_factor
                    return p_of_n
                #Ns = self.elements
                self.probabilities = exp_distr(self.elements) # compute prob of each element
            p_cum = np.cumsum(self.probabilities)
            return sample_from_discrete_distr(n_samples, p_cum)
        else:
            if self.distr == 'uniform':
                samples = np.random.rand(n_samples)*(self.x_max - self.x_min) - self.x_min
            else:
                u = np.random.rand(n_samples)*(np.log(self.x_max)-np.log(self.x_min)) + np.log(self.x_min)
                samples = np.exp(u)
            return samples

save = True
n_runs = 4
n_episodes = 400
HP_scores, HP_asymptotic_score, HP_asymptotic_std = HP_Search(n_runs, n_episodes, list_of_dict)

if save:
    np.save("HP_scores", HP_scores)
    np.save("HP_asymptotic_score", HP_asymptotic_score)
    np.save("HP_asymptotic_std", HP_asymptotic_std)
