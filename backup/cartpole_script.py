import numpy as np
import matplotlib.pyplot as plt
import torch
import gym
import agents

def train_cartpole(n_episodes = 100, lr = 0.01, gamma = 0.99, debug=False):
    dprint = print if debug else lambda *args, **kwargs : None
    # Create environment
    env = gym.make("CartPole-v1")
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    # Init agent
    agent = agents.PolicyGrad(observation_space, action_space, lr, gamma)
    performance = []
    for e in range(n_episodes):
        # Reset environment (start of an episode)
        state = env.reset()
        rewards = []
        log_probs = []
        
        while True:
            action, log_prob = agent.get_action(state, return_log = True, greedy=False)
            dprint("action: ", action)
            dprint("log_prob ", log_prob)
            new_state, reward, terminal, info = env.step(action) # gym standard step's output
            dprint("terminal ", terminal)
            rewards.append(reward)
            log_probs.append(log_prob)
            if terminal:
                break

            state = new_state
        print("Episode %d - reward: %.0f"%(e+1, np.sum(rewards)))
        agent.update(rewards, log_probs)
        performance.append(np.sum(rewards))
    return agent, np.array(performance)

trained_agent, cumulative_reward = train_cartpole(n_episodes = 100, debug=False)

episodes = np.arange(1,len(cumulative_reward)+1)
plt.plot(episodes, cumulative_reward)
plt.close()

def render_test_episode(agent):
    # Create environment
    env = gym.make("CartPole-v1")
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    state = env.reset()
    while True:
        env.render()
        action = agent.get_action(state, return_log = False, greedy=False)
        new_state, reward, terminal, info = env.step(action) # gym standard step's output
        if terminal: 
            break
        else: 
            state = new_state
    return
           
render_test_episode(trained_agent)
