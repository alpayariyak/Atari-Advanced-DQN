#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH -t 20:00:00
#SBATCH --mem 50G

python -u main.py --learning_rate=0.0003 --epsilon_end=0.01 --train_dqn --test_n=16 --n_episodes=3000000 --buffer_size=600000 --load_checkpoint=1 > results/test16_double.out