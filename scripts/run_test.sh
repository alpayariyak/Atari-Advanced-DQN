#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 0:3:00
#SBATCH --mem 5G

python -u main.py --test_dqn > test_out16-5.out