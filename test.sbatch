#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=test.out
#SBATCH --nodes=40
#SBATCH --ntasks=100
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr
#SBATCH --account=macs30123

module load python/anaconda-2019.03
#module load python/anaconda-2021.05
#module load pytorch/1.2
#tensorflow/2.1
#module load cuda
module load intelmpi/2018.2.199+intel-18.0

mpirun -n 100 python3 process.py > test.out
