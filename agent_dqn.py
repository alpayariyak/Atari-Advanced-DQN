#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN
from ExperienceBuffer import *

"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN, self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #

        self.buffer_size = args.buffer_size
        self.minibatch_size = args.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer = ExperienceBuffer(self.buffer_size, self.minibatch_size, self.device)
        self.n_episodes = args.n_episodes
        self.gamma = args.gamma

        self.epsilon_range = (args.epsilon_start, args.epsilon_end)
        self.epsilon = self.epsilon_range[0]
        self.decay_start = args.decay_start
        self.decay_end = args.decay_end if args.decay_end else int(self.n_episodes / 2)
        self.epsilon_stepsize = (self.epsilon_range[0] - self.epsilon_range[1]) / (self.decay_end - self.decay_start)

        self.Q_network = DQN(self.device, initialize_weights=args.initialize_weights)
        self.Q_target_network = DQN(self.device).eval()
        self.update_target()

        self.learning_rate = args.learning_rate
        self.loss = torch.nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=self.learning_rate)

        self.current_episode = 0

        self.optimize_interval = args.optimize_interval
        self.target_update_interval = args.target_update_interval  # (self.n_episodes * 0.01)
        self.evaluate_interval = args.evaluate_interval  # (self.n_episodes * 0.1)

        self.loss_list = []
        self.rewards_list = []
        self.action_counter = {0: 0, 1: 0, 2: 0, 3: 0}

        if args.test_dqn:
            # you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #

    def update_target(self):
        self.Q_target_network.load_state_dict(self.Q_network.state_dict())

    def update_epsilon(self):
        if self.current_episode > self.decay_start:
            self.epsilon = max(self.epsilon - self.epsilon_stepsize, self.epsilon_range[1])

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        ###########################
        pass

    def make_action(self, observation, test=False):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if random.random() < 1 - self.epsilon or test:
            with torch.no_grad():
                action = self.Q_network(torch.from_numpy(observation).unsqueeze(0)).argmax().item()
                self.action_counter[action] += 1
        else:
            action = self.env.action_space.sample()
        ###########################
        return action

    def run_episode(self, test=False):
        state = self.env.reset()
        terminated, truncated = False, False
        episode_reward, episode_length = 0, 0

        while not terminated:
            action = self.make_action(state, test)
            new_state, reward, terminated, truncated, _ = self.env.step(action)
            if not test:
                self.buffer.push((state, action, reward, new_state, terminated))
            episode_length += 1
            episode_reward += reward
            state = new_state
        return episode_reward, episode_length

    def fill_buffer(self):
        while not self.buffer.is_full():
            self.run_episode()

    def optimize(self):
        samples = self.buffer.sample_experiences()
        states, actions, rewards, next_states, terminals = [list_to_tensor(sample) for sample in samples]
        actions, rewards, terminals = actions.unsqueeze(1), rewards.unsqueeze(1), terminals.unsqueeze(1)
        Q_values = self.Q_network(states).gather(1, actions)
        max_Q_target_next_states = self.Q_target_network(next_states).max(1)[0].view(self.minibatch_size, 1)
        Q_target_values = rewards + self.gamma * max_Q_target_next_states * (1 - terminals.long())
        loss = self.loss(Q_values, Q_target_values.detach())
        self.loss_list.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.Q_network.parameters(), 1.0)
        self.optimizer.step()

    def train(self):
        """
        Implement your training algorithm here
        """
        self.fill_buffer()
        avg_last_30_ep_rewards = 0

        for episode in range(self.n_episodes):
            self.current_episode = episode
            episode_reward, episode_length = self.run_episode()
            avg_last_30_ep_rewards += episode_reward/30

            if episode % self.optimize_interval == 0:
                self.optimize()
            if episode % self.target_update_interval == 0:
                self.update_target()
            if episode % self.evaluate_interval == 0:
                self.evaluate()

            if episode % 30 == 0 and episode != 0:
                avg_last_30_ep_rewards = 0
                self.rewards_list.append(avg_last_30_ep_rewards)

            self.update_epsilon()

    def evaluate(self, eval_episodes=100):
        self.action_counter = {0: 0, 1: 0, 2: 0, 3: 0}
        episode_rewards = 0
        episode_lengths = 0
        self.Q_network.eval()
        for episode in range(eval_episodes):
            episode_reward, episode_length = self.run_episode(test=True)
            episode_rewards += episode_reward
            episode_lengths += episode_length
        self.Q_network.train()

        print(f"Episode: {self.current_episode}  Reward: {episode_rewards / eval_episodes}")
        print(f"Epsilon: {self.epsilon}  Last 5 Losses: {self.loss_list[-5:]}")
        print(self.action_counter)
        print(f"Ep length: {episode_lengths / eval_episodes}\n")


def list_to_tensor(a_list):
    return torch.from_numpy(np.stack(a_list))
