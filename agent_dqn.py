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

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """Initialize the DQN agent.

        Argumetns:
            env: the environment the agent will interact with
            args: a namespace object containing hyperparameters and other arguments
        """
        super(Agent_DQN, self).__init__(env)
        self.buffer_size = args.buffer_size  # size of experience buffer
        self.minibatch_size = args.batch_size  # size of minibatch to sample from buffer
        self.device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")  # choose device (CPU or GPU) for PyTorch
        self.buffer = ExperienceBuffer(self.buffer_size, self.minibatch_size, self.device)  # create experience buffer
        self.n_episodes = args.n_episodes  # number of episodes to run
        self.gamma = args.gamma  # discount factor

        # initialize epsilon for exploration-exploitation trade-off
        self.epsilon_range = (args.epsilon_start, args.epsilon_end)
        self.epsilon = self.epsilon_range[0]
        self.decay_start = args.decay_start  # episode at which to start decreasing epsilon
        self.decay_end = args.decay_end if args.decay_end else int(self.n_episodes / 2) # episode at which to end decreasing epsilon (defaults to n_episodes / 2)
        self.epsilon_stepsize = (self.epsilon_range[0] - self.epsilon_range[1]) / (self.decay_end - self.decay_start) # step size for decreasing epsilon

        self.Q_network = DQN(self.device, initialize_weights=args.initialize_weights)  # create Q network
        self.Q_target_network = DQN(self.device).eval()  # create target network (in eval mode)
        self.update_target()  # initialize target network with same weights as Q network

        self.learning_rate = args.learning_rate  # learning rate for optimizer
        self.loss = torch.nn.SmoothL1Loss()  # loss function
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=args.learning_rate, eps=1.5e-4)  # optimizer

        self.current_episode = 0  # counter for current episode

        self.optimize_interval = args.optimize_interval  # number of timesteps between model optimization
        self.target_update_interval = args.target_update_interval  # (self.n_episodes * 0.01)
        self.evaluate_interval = args.evaluate_interval  # (self.n_episodes * 0.1)

        self.clip_grad = args.clip_grad # whether to clip gradients

        self.loss_list = [] # list to store loss values
        self.rewards_list = [] # list to store rewards
        self.action_counter = {0: 0, 1: 0, 2: 0, 3: 0} # dictionary to store action counts

        self.test_n = args.test_n # test number

        if args.test_dqn:
            # load model
            print('loading trained model')
            self.Q_network.load_state_dict(torch.load(args.path_to_trained_model, map_location=self.device))
            self.Q_network.eval()

        if args.load_checkpoint != False:
            # load model checkpoint
            self.Q_network.load_state_dict(
                torch.load(f'checkpoints/test{args.load_checkpoint}.pt', map_location=self.device))
            self.optimizer = optim.Adam(self.Q_network.parameters(), lr=args.learning_rate, eps=1.5e-4)
            self.update_target()
            self.epsilon_stepsize = 0
            self.epsilon = 0.01

    def update_target(self):
        # update target network with weights from Q network
        self.Q_target_network.load_state_dict(self.Q_network.state_dict())

    def update_epsilon(self):
        # update epsilon for exploration-exploitation trade-off
        if self.current_episode > self.decay_start:
            self.epsilon = max(self.epsilon - self.epsilon_stepsize, self.epsilon_range[1])

    def make_action(self, observation, test=False):
        """
        Return predicted action of your agent with epsilon-greedy policy.
        Arguments:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        if random.random() < 1 - self.epsilon or test:
            # use Q network to predict action
            with torch.no_grad():
                action = self.Q_network(torch.from_numpy(observation).unsqueeze(0)).argmax().item()
                self.action_counter[action] += 1
        else:
            # choose random action
            action = self.env.action_space.sample()
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
        # fill experience buffer with random actions
        while not self.buffer.is_full():
            self.run_episode()

    def optimize(self):
        # sample minibatch from experience buffer
        samples = self.buffer.sample_experiences()
        states, actions, rewards, next_states, terminals = [list_to_tensor(sample).to(self.device) for sample in
                                                            samples]
        actions, rewards, terminals = actions.unsqueeze(1), rewards.unsqueeze(1), terminals.unsqueeze(1)

        # calculate Q and target values for minibatch
        Q_values = self.Q_network(states).gather(1, actions)
        max_Q_target_next_states = self.Q_target_network(next_states).max(1)[0].view(self.minibatch_size, 1).detach()
        Q_target_values = rewards + self.gamma * max_Q_target_next_states * (1 - terminals.long())

        # calculate loss
        loss = self.loss(Q_values, Q_target_values)

        # optimize model
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.Q_network.parameters(), 1.0)
        self.optimizer.step()

    def train(self):

        # fill experience buffer
        self.fill_buffer()

        for episode in range(self.n_episodes):
            self.current_episode = episode

            self.run_episode()

            if episode % self.optimize_interval == 0:
                self.optimize()
            if episode % self.target_update_interval == 0:
                self.update_target()
            if episode % self.evaluate_interval == 0:
                self.evaluate()

            if episode % 10000 == 0:
                if self.test_n:
                    torch.save(self.Q_network.state_dict(), f'checkpoints/test{self.test_n}.pt')

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
        print(f"Epsilon: {self.epsilon}")
        print(self.action_counter)
        print(f"Ep length: {episode_lengths / eval_episodes}\n")


def list_to_tensor(a_list):
    return torch.from_numpy(np.stack(a_list))
