#!/bin/sh
#SBATCH --time=3-00:00:00
#SBATCH --job-name=CS5239
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1124485@comp.nus.edu.sg
#SBATCH --partition=long


gcc -O3 mat_mul2.c -o mat_mul1
gcc -O3 mat_mul2.c -o mat_mul2
gcc -mavx512dq -mavx512f -O3 my_mat_mul.c -o my_mat_mul

python test.py
