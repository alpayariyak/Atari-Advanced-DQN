#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -C A100
#SBATCH --mem 60G

python -u main.py --decay_end=1 --target_update_interval=5000 --learning_rate=0.0003 --epsilon_end=0.01 --train_dqn --test_n=17 --n_episodes=5000000 --buffer_size=1000000 > results/test17_d.out
python -u main.py --decay_end=1 --target_update_interval=5000 --learning_rate=0.0005 --epsilon_end=0.01 --train_dqn --test_n=18 --n_episodes=5000000 --buffer_size=1000000 > results/test18_d.out
python -u main.py --decay_end=1 --target_update_interval=5000 --learning_rate=0.0009 --epsilon_end=0.01 --train_dqn --test_n=19 --n_episodes=5000000 --buffer_size=1000000 > results/test19_d.out