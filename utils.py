import numpy as np
import matplotlib.pyplot as plt
import time
import environment

def render(agent, save=False, x=10, y=10, goal=[9,9], initial=[0,0], greedy=False):
    fig = plt.figure(figsize = (8,6))
    # initialize environment
    env = environment.Sandbox(x, y, initial, goal, max_steps=50)
    reward = 0

    # 
    rgb_map = np.full((x,y,3), [199,234,70])/255.
    rgb_map[goal[0],goal[1],:] = np.array([255,255,255])/255.
    rgb_map[initial[0],initial[1],:] = np.array([225,30,100])/255.
    plt.imshow(rgb_map) # show map
    plt.title("Sandbox Env - Turn: %d"%(0))
    plt.yticks([])
    plt.xticks([])
    fig.show()
    time.sleep(0.75) #uncomment to slow down for visualization purposes
    if save:
        plt.savefig('.raw_gif/turn%.3d.png'%0)

    # run episode
    state = env.reset()
    for step in range(0, env.max_steps):
        state = np.array([state])
        action, log_prob = agent.get_action(state, return_log = True, greedy=greedy)
        new_state, reward, terminal, info = env.step(action) # gym standard step's output

        plt.cla() # clear current axis from previous drawings -> prevents matplotlib from slowing down
        rgb_map = np.full((x,y,3), [199,234,70])/255.
        rgb_map[goal[0],goal[1],:] = np.array([255,255,255])/255.
        rgb_map[env.state[0],env.state[1],:] = np.array([225,30,100])/255.
        plt.imshow(rgb_map)
        plt.title("Sandbox Env - Turn: %d "%(step+1))
        plt.yticks([]) # remove y ticks
        plt.xticks([]) # remove x ticks
        fig.canvas.draw() # update the figure
        time.sleep(0.75) #uncomment to slow down for visualization purposes
        if save:
            plt.savefig('.raw_gif/turn%.3d.png'%(step+1))

        if terminal:
            break
        state = new_state
        
    return