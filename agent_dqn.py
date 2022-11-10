#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys

import torch

import torch.nn.functional as F
from torch import nn
import torch.optim as optim

from agent import Agent
from dqn_model import DQN
from ExperienceBuffers import *

"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


# 1 gpu 8 ram 4 cpu
class Agent_DQN(Agent):
    def __init__(self, env, args, buffer=ExperienceBuffer):
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

        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.n_episodes = args.n_episodes
        self.current_episode = 0

        self.epsilon_range = (args.epsilon_start, args.epsilon_end)
        self.epsilon = self.epsilon_range[0]
        self.decay_start = args.decay_start if args.decay_start else 0
        self.decay_end = args.decay_end if args.decay_end else int(self.n_episodes / 2)
        self.epsilon_stepsize = (self.epsilon_range[0] - self.epsilon_range[1]) / (self.decay_end - self.decay_start)

        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.buffer = buffer(self.buffer_size, self.batch_size, self.device)

        self.Q_net = DQN(self.device)
        self.Q_target_net = DQN(self.device).eval()
        self.update_target()

        self.learning_rate = args.learning_rate
        self.gamma = args.gamma
        self.loss = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.Q_net.parameters(), lr=self.learning_rate)
        self.grad_clip =args.grad_clip

        self.optimize_interval = 4 if not args.optimize_interval else args.optimize_interval
        self.target_update_interval = 5000 if not args.target_update_interval else args.target_update_interval # (self.n_episodes * 0.01)
        self.evaluate_interval = 10000 if not args.evaluate_interval else args.evaluate_interval# (self.n_episodes * 0.1)

        self.episode_rewards = []
        self.loss_list = []
        self.action_counter = {0:0, 1:0, 2:0, 3:0}

        if args.checkpoint_name:
            print(args.checkpoint_name)
            checkpoint = torch.load(f'checkpoints/{args.checkpoint_name}.pt', map_location=torch.device('cpu') )
            self.Q_net.load_state_dict(checkpoint['model_state_dict'])

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.update_target()
            self.epsilon = 0.1

        if args.test_dqn:
            # you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            self.Q_net.eval()

    def update_epsilon(self):
        if self.current_episode > self.decay_start:
            self.epsilon = max(self.epsilon - self.epsilon_stepsize, self.epsilon_range[1])

    def update_target(self):
        self.Q_target_net.load_state_dict(self.Q_net.state_dict())

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

    def make_action(self, observation, test=False, eps=False):
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
        if eps:
            epsilon = eps
        else:
            epsilon = self.epsilon
        
        if random.random() < 1 - epsilon or test:
            with torch.no_grad():
                if test:
                    observation = torch.from_numpy(observation).to(self.device)
                action = self.Q_net(observation.unsqueeze(0)).argmax().item()
                self.action_counter[action] += 1
        else:
            print(1)
            action = self.env.action_space.sample()
        ###########################
        return action

    def optimize(self):
        states, actions, rewards, next_states, terminals = self.buffer.sample_experiences()

        Q_values = self.Q_net(states).gather(1, actions.reshape(self.batch_size, 1))
        next_states_max_Q_target = self.Q_target_net(next_states).max(1)[0]
        Q_target_values = rewards + self.gamma * next_states_max_Q_target * (1 - terminals.long())

        loss = self.loss(Q_target_values.unsqueeze(1).detach(), Q_values)
        self.loss_list.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        # clip gradients to (-1,1)
        torch.nn.utils.clip_grad_norm_(self.Q_net.parameters(), 1.0)
        self.optimizer.step()

    def to_tensor(self, value):
        return torch.from_numpy(value).to(self.device)

    def run_episode(self, eval_mode=False, test=False):
        state = self.to_tensor(self.env.reset())
        terminated, truncated = False, False
        episode_reward = 0
        episode_length = 0
        while not terminated:
            action = self.make_action(state, test)
            new_state, reward, terminated, truncated, _ = self.env.step(action)
            new_state = self.to_tensor(new_state)
            if not eval_mode:
                self.buffer.push(state, action, reward, new_state, terminated)
            state = new_state
            episode_reward += reward
            episode_length += 1
        return episode_reward, episode_length

    def fill_buffer(self):
        while self.buffer.current_buffer_length < self.buffer_size:
            self.run_episode()
            print(self.buffer.current_buffer_length)

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        self.fill_buffer()

        for episode in range(self.n_episodes):
            episode_reward = 0.0
            self.current_episode = episode

            self.run_episode()

            if episode % self.optimize_interval == 0:
                self.optimize()
            if episode % self.target_update_interval == 0:
                self.update_target()
            if episode % self.evaluate_interval == 0:
                self.evaluate()
            if episode % self.n_episodes * 0.1 == 0:
                torch.save({'episode': episode,
                'model_state_dict': self.Q_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.loss_list[-1]}, f'checkpoints/{self.decay_end}.pt')
            self.update_epsilon()
        ###########################

    def evaluate(self, eps=0.05, val_size=5):
        self.action_counter = {0:0, 1:0, 2:0, 3:0}
        episode_rewards = 0
        episode_lengths = 0
        self.Q_net.eval()
        for episode in range(val_size):
            episode_reward, episode_length = self.run_episode(eval_mode=True, test=True)
            episode_rewards += episode_reward
            episode_lengths += episode_length
        self.Q_net.train()

        print(f"Episode: {self.current_episode}  Reward: {episode_rewards / val_size}")
        print(f"Epsilon: {self.epsilon}  Last Loss: {self.loss_list[-1]}")
        print(self.action_counter)
        print(f"Ep length: {episode_lengths / val_size}\n")


    def test_epsilon(self):
        print(f'Decay end episode: {self.decay_end}')
        for episode in range(self.n_episodes):
            self.current_episode = episode
            self.update_epsilon()
            if self.epsilon == self.epsilon_range[1]:
                print(f"Actual end: {episode}")
                break
