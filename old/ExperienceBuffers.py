from collections import deque, namedtuple
import random
import numpy as np
import torch
from operator import itemgetter

class PrioritizedExperienceBuffer:
    # How To Speed Up Training With Prioritized Experience Replay - https://www.youtube.com/watch?v=MqZmwQoOXw4
    def __init__(self, maxlen):
        self.experience_buffer = deque(maxlen=maxlen)  # (s, a, r, s', is_terminal)
        self.priorities = deque(maxlen=maxlen)  # (index, p_i, P_i, w_i)

    def push(self):
        """ You can add additional arguments as you need.
        Push new data to buffer and remove the old one if the buffer is full.

        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        ###########################

    def select_experience(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        ###########################
        return


class ExperienceBuffer:
    def __init__(self, maxlen, minibatch_size, device):
        self.states = deque(maxlen=maxlen)
        self.actions = deque(maxlen=maxlen)
        self.rewards = deque(maxlen=maxlen)
        self.next_states = deque(maxlen=maxlen)
        self.terminal_buffer = deque(maxlen=maxlen)

        self.device = device

        self.minibatch_size = minibatch_size
        self.max_buffer_capacity = maxlen
        self.current_buffer_length = 0

    def push(self, state, action, reward, next_state, terminal):  # (s, a, r, s', is_terminal)
        """ You can add additional arguments as you need.
        Push new data to buffer and remove the old one if the buffer is full.

        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.current_buffer_length += 1
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.terminal_buffer.append(terminal)

        ###########################

    def sample_experiences(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        buffers_non_state = [self.actions, self.rewards, self.terminal_buffer]
        buffers_state = [self.states, self.next_states]
        sample_indices = random.sample(range(self.max_buffer_capacity), self.minibatch_size)
        actions, rewards, terminals = [torch.tensor(list(itemgetter(*sample_indices)(buffer))).to(self.device) for buffer in buffers_non_state]
        states, next_states = [torch.stack(list(itemgetter(*sample_indices)(buffer))).to(self.device) for buffer in buffers_state]
        return states.float(), actions, rewards.float(), next_states.float(), terminals

