#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=test.out
#SBATCH --mem-per-cpu=8G
#SBATCH --nodes=40
#SBATCH --ntasks=40
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr
#SBATCH --account=macs30123

module load python/anaconda-2019.03
#module load python/anaconda-2021.05
#module load pytorch/1.2
#tensorflow/2.1
#module load cuda
module load intelmpi/2018.2.199+intel-18.0

rm -f test.out
mpirun -n 40 python3 process.py > test.out
