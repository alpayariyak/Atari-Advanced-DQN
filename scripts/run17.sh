#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 20:00:00
#SBATCH -C A100
#SBATCH --mem 30G

python -u main.py --target_update_interval=5000 --learning_rate=0.000025 --epsilon_end=0.01 --train_dqn --test_n=17 --n_episodes=5000000 --buffer_size=500000 > results/test17_d.out