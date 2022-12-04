#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch


class DQN(nn.Module):
    """
    Initialize a deep Q-learning network

    Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    Args:
        device: The device on which to run the model.
        initialize_weights: (bool) Whether to initialize the model weights.
        in_channels: (int) The number of channels in the input image.
        num_actions: (int) The number of possible actions in the environment.
    """

    def __init__(self, device, initialize_weights=False, in_channels=4, num_actions=4):
        super(DQN, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

        if initialize_weights:
            self.initialize_weights()

        self.to(self.device)

    def forward(self, x):
        # Using the original Deepmind architecture
        x = x.permute(0, 3, 1, 2).float().to(self.device) / 255.0  # Normalization
        # CNN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Linear layers
        x = F.relu(self.fc1(torch.flatten(x, start_dim=1)))
        x = self.fc2(x)
        ###########################
        return x

    def initialize_weights(self):
        # Weight initialization modified and inspired from https://github.com/jasonbian97/Deep-Q-Learning-Atari-Pytorch
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
