import numpy as np
import random
from collections import deque

class ValueHistory:

    def __init__(self, memory_size):
        self.buffer = deque()
        self.memory_size = memory_size

    def append(self, state, V):
        self.buffer.append((state, V))
        if len(self.buffer) >= self.memory_size:
            self.buffer.popleft()

    def sample(self, size):
        minibatch = random.sample(self.buffer, size)
        states = np.array([data[0] for data in minibatch])
        Vs = np.array([data[1] for data in minibatch])
        
        return states, Vs