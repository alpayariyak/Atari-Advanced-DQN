from collections import deque, namedtuple
import random
import numpy as np
import torch
from operator import itemgetter


class ExperienceBuffer:
    def __init__(self, maxlen, minibatch_size, device):
        self.max_buffer_capacity = maxlen
        self.minibatch_size = minibatch_size
        self.buffer = deque(maxlen=maxlen)
        self.device = device

    def is_full(self):
        return len(self.buffer) == self.max_buffer_capacity

    def push(self, experience_tuple):  # (s, a, r, s', is_terminal)
        self.buffer.append(experience_tuple)

    def sample_experiences(self):
        """
        Select batch from buffer.
        """
        sampled_tuples = random.sample(self.buffer, self.minibatch_size)
        states, actions, rewards, next_states, terminals = [], [], [], [], []

        for (state, action, reward, next_state, terminal) in sampled_tuples:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            terminals.append(terminal)

        return [states, actions, rewards, next_states, terminals]
