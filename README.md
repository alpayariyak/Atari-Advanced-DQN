# Breakout Deep Q-Network

This project is an implementation of the deep Q-learning (DQN) algorithm. The DQN algorithm is a model-free reinforcement learning technique that can be used to learn a policy for taking actions in an environment in order to maximize a reward signal.
<img src="https://miro.medium.com/max/640/1*jXpSVhjWRxgzDDKKVAQR8A.gif" width="250" height="300" />

## Getting Started

### Prerequisites

To run this project, you will need to have the following packages installed:

- Python 3.6 or later
- NumPy
- PyTorch
- OpenAI Gym
- OpenCV-Python

You can install these packages using `pip`:
```
pip install -r requirements.txt
```

### Running the Code
To evaluate a trained model, run the `main.py` script with the '--test' flag and the path to the model checkpoint:
```
python main.py --test --model-path <path to model>
```

To train and evaluate the DQN agent, run the `main.py` script with the `--train` flag:
```
python main.py --train
```

This will train the DQN agent on the `AtariBreakoutNoFrameskip-v4` environment from the OpenAI Gym for a specified number of episodes, and will save checkpoints of the trained model at every 10,000 episodes. The model checkpoint will be saved in the `checkpoints` directory.

## Code Overview

- `agent.py`: This file defines the `Agent` class, which is an abstract base class for DQN agents.
- `agent_dqn.py`: This file defines the Agent_DQN class, which extends the Agent class and implements the DQN algorithm. The Agent_DQN class has methods for interacting with an environment, storing and sampling from past experiences, training the DQN model, and evaluating the trained model.
- `argument.py`: This file defines a Argument class that is used to parse and store command-line arguments passed to the main.py script.
- `atari_wrapper.py`: This file defines the AtariWrapper class, which is a wrapper for the OpenAI Gym Atari environments. The AtariWrapper class preprocesses raw frames from the environment and converts them to grayscale and resizes them.
- `dqn_model.py`: This file defines the DQN class, which is a PyTorch neural network that can be used to approximate the Q-function for the DQN algorithm.
- `environment.py`: This file defines the Environment class, which is a wrapper for an OpenAI Gym environment. The Environment class has methods for resetting the environment, stepping through the environment, and rendering the environment.
- `ExperienceBuffer.py`: This file defines the ExperienceBuffer class, which is used to store and sample from past experiences in the environment. The ExperienceBuffer class has methods for adding experiences to the buffer, sampling a minibatch of experiences from the buffer, and updating the priority of experiences in the buffer.
- `main.py`: This is the main script for the project. It uses the classes and methods defined in the other files to train and evaluate a DQN agent on the Atari Breakout game with no frameskips.
- `requirement.txt`: This file contains the list of packages required to run the code in this project.
- `test.py`: This file contains code to evaluate a trained model.

## Hyperparameters and Other Arguments

The behavior of the DQN agent can be customized by passing command-line arguments to the main.py script. The available arguments and their default values are defined in the argument.py file. Some of the key arguments are:

- `--buffer_size`: the size of the experience buffer (default: 100000)
- `--batch_size`: the size of the minibatches to sample from the experience buffer (default: 32)
- `--n_episodes`: the number of episodes to run (default: 1000)
- `--gamma`: the discount factor (default: 0.99)
- `--epsilon_start`: the initial value of epsilon for the exploration-exploitation trade-off (default: 1.0)
- `--epsilon_end`: the final value of epsilon after the exploration-exploitation trade-off has decayed (default: 0.01)
- `--decay_start`: the episode at which to start decreasing epsilon (default: 100)
- `--decay_end`: the episode at which to end decreasing epsilon (default: None, which defaults to n_episodes / 2)
- `--learning_rate`: the learning rate for the optimizer (default: 0.0001)
- `--optimize_interval`: the number of timesteps between model optimization (default: 4)
- `--target_update_interval`: the number of timesteps between updates to the target network (default: 10000)
- `--evaluate_interval`: the number of episodes between evaluations of the trained model (default: 100)

## Results
The performance of the DQN agent on the Atari Breakout game with no frameskips will vary depending on the hyperparameters and other arguments used. With hyperparameter tuning the agent in this project was able to achieve an average score of 70 points over 100 games. This is higher than the initial score of around 40 reported by DeepMind in their original DQN paper.
