import argparse
from test import test
from environment import Environment
import time


def parse():
    """
    Parse command line arguments and return an `argparse.Namespace` object.
    """
    parser = argparse.ArgumentParser(description="Breakout")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def run(args):
    """
    Execute the main program that can either train or test an agent.
    """
    start_time = time.time()
    if args.train_dqn:
        env_name = args.env_name or 'BreakoutNoFrameskip-v4'
        env = Environment(env_name, args, atari_wrapper=True, test=False)
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        agent.train()

    if args.test_dqn:
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=100, record_video=False)
    print('running time:', time.time() - start_time)


if __name__ == '__main__':
    args = parse()
    run(args)
