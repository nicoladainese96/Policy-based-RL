import numpy as np
import time

class Sandbox():
    
    def __init__(self, x, y, initial, goal, R0=0, max_steps=0):
        self.boundary = np.asarray([x, y])
        self.initial = np.asarray(initial)
        self.state = np.asarray(initial)
        self.goal = goal
        self.R0 = R0
        # the agent makes an action (0 is up, 1 is down, 2 is right, 3 is left)
        self.action_map = {
                            0: [0, 1],
                            1: [0, -1],
                            2: [1, 0],
                            3: [-1, 0],
                          }
        self.max_steps = 5*int(np.max([x,y])) # 5 times the greatest linear dimension
        self.current_steps = 0
        
    def step(self, action):
        self.current_steps += 1
        # Baseline reward
        reward = self.R0 
        # Get grid movement
        movement = self.action_map[action]
        # Compute next vectorial state
        next_state = self.state + np.asarray(movement)
        if(self.check_boundaries(next_state)):
            # Enforce staying within boundaries with negative reward
            reward = -1
        else:
            # Update state only if valid movement
            self.state = next_state
            
        if (self.state == self.goal).all():
            reward = 1
            terminal = True
        else:
            terminal = False
        
        # Check if number of steps has exceeded the maximum for an episode
        info = {}
        if self.current_steps == self.max_steps:
            terminal = True
            info['TimeLimit.truncated'] = True
            
        enc_state = self.encode_state()
        return enc_state, reward, terminal, info

    # map action index to movement
    def check_boundaries(self, state):
        out = len([num for num in state if num < 0])
        out += len([num for num in (self.boundary - np.asarray(state)) if num <= 0])
        return out > 0

    def encode_state(self):
        # encode row by row
        # e = X*y + x
        enc_state = self.boundary[0]*self.state[1] + self.state[0]
        return enc_state
    

    def reset(self):
        self.state = self.initial
        self.current_steps = 0
        return self.encode_state()

