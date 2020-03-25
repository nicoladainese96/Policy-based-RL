import numpy as np
import torch
import gym
import ActorCritic
import time

def play_episode(agent, env, return_states=False, shape_r=True, bootstrap_flag=False):

    # Reset environment (start of an episode)
    state = env.reset()
    rewards = []
    log_probs = []
    done = []
    
    if return_states:
        states = [state]
    if bootstrap_flag:
        bootstrap = []
        
    steps = 0
    while True:
        steps += 1
        action, log_prob = agent.get_action(state, return_log = True)
        new_state, reward, terminal, info = env.step(action) # gym standard step's output
        
        if return_states:
            states.append(new_state)
        
        # See if bootstrap is needed
        
        if bootstrap_flag:
            if terminal and 'TimeLimit.truncated' in info:
                bootstrap.append(True)
            else:
                bootstrap.append(False)
        
        # See if reward shaping is needed
        
        if shape_r:
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
    bootstrap = np.array(bootstrap)
    
    if bootstrap_flag:
        if return_states:
            return rewards, log_probs, np.array(states), done, bootstrap
        else:
            return rewards, log_probs, done, bootstrap
    else:
        if return_states:
            return rewards, log_probs, np.array(states), done
        else:
            return rewards, log_probs, done

def train_cartpole_A2C(agent, env, n_episodes = 1000, shape_r=True, bootstrap_flag=False):
    performance = []
    for e in range(n_episodes):
        
        if bootstrap_flag:
            rewards, log_probs, states, done, bootstrap = play_episode(agent, env, True, shape_r, bootstrap_flag)
        else:
            rewards, log_probs, states, done = play_episode(agent, env, True, shape_r, bootstrap_flag)
            
        performance.append(np.sum(rewards))
        if (e+1)%100 == 0:
            print("Episode %d - reward: %.0f"%(e+1, np.mean(performance[-100:])))
        if bootstrap_flag:
            agent.update(rewards, log_probs, np.array([states]), done, bootstrap)
        else:
            agent.update(rewards, log_probs, np.array([states]), done)
            
    performance = np.array(performance)
    L = n_episodes // 6
    return performance, performance[-L:].mean(), performance[-L:].std()/np.sqrt(L)

def evaluate_agent(n_runs, n_episodes, shape_r=True, bootstrap_flag=False, **HPs):
    runs_scores = []
    asymptotic_score = []
    asymptotic_std = []
    
    print("shape_r", shape_r)
    print("bootstrap_flag", bootstrap_flag)
    
    for r in range(n_runs):
        
        print("\nRun %d/%d: "%(r+1, n_runs)) 
        
        # Create environment
    
        env = gym.make("CartPole-v1")
        observation_space = env.observation_space.shape[0]
        action_space = env.action_space.n
    
        # Init agent
    
        agent = ActorCritic.A2C(observation_space, action_space, **HPs)
    
        # Evaluate agent on a single run 
        
        perf, asymptotic_perf, asymptotic_err = train_cartpole_A2C(agent, env, n_episodes, shape_r, bootstrap_flag)

        runs_scores.append(perf)
        asymptotic_score.append(asymptotic_perf)
        asymptotic_std.append(asymptotic_err)
        
    return np.array(runs_scores), np.mean(asymptotic_score), np.mean(asymptotic_std)

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
    
def HP_Search(n_runs, n_episodes, list_of_HP_dict, shape_r=False, bootstrap_flag=True):
    
    HP_scores = []
    HP_asymptotic_score = []
    HP_asymptotic_std = []
    start = time.time()
    
    for i, HP in enumerate(list_of_HP_dict):
        
        print("\nEvaluating HP %d / %d... "%(i+1, len(list_of_HP_dict)))
        runs_scores, asymptotic_score, asymptotic_std = evaluate_agent(n_runs, n_episodes, shape_r, bootstrap_flag, **HP, )
        print_HP_score(HP, asymptotic_score, asymptotic_std)
        HP_scores.append(runs_scores)
        HP_asymptotic_score.append(asymptotic_score)
        HP_asymptotic_std.append(asymptotic_std)
        elapsed_min = (time.time() - start)/60
        print("Evaluated  HP %d / %d - took %.2f min."%(i+1, len(list_of_HP_dict), elapsed_min))
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