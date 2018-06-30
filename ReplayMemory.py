import random
from collections import deque
import numpy as np

class ReplayMemory:

    def __init__(self, memory_size):
        self.buffer = deque()
        self.memory_size = memory_size

    def append(self, pre_state, action, reward, post_state, done):
        self.buffer.append((pre_state, action, reward, post_state, done))
        if len(self.buffer) >= self.memory_size:
            self.buffer.popleft()

    def sample(self, size):
        minibatch = random.sample(self.buffer, size)
        states = np.array([data[0] for data in minibatch])
        actions = np.array([data[1] for data in minibatch])
        rewards = np.array([data[2] for data in minibatch])
        next_states = np.array([data[3] for data in minibatch])
        dones = np.array([data[4] for data in minibatch])
        
        return states, actions, rewards, next_states, dones
