#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --mem 30G

python -u main.py --decay_end=1 --target_update_interval=5000 --learning_rate=0.0003 --epsilon_end=0.01 --train_dqn --test_n=27 --n_episodes=1000000 --buffer_size=100000 > results/testfinaldouble2.out