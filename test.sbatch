#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=test.out
#SBATCH --ntasks=2
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr
#SBATCH --account=macs30123

module load python/anaconda-2019.03
module load intelmpi/2018.2.199+intel-18.0

mpirun -n 2 python3 test.py > test.out
