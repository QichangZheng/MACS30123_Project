#!/bin/bash
#SBATCH --mem-per-cpu=8G    # memory per CPU core
#SBATCH --job-name=test      # job name
#SBATCH --output=test.out    # output log file
#SBATCH --error=gpu.err     # error file
#SBATCH --time=00:05:00     # 5 minutes of wall time
#SBATCH --nodes=2          # 1 GPU node
#SBATCH --partition=gpu2    # GPU2 partition
#SBATCH --ntasks=2          # 1 CPU core to drive GPU
#SBATCH --gres=gpu:2        # Request 1 GPU
#SBATCH --account=macs30123 # specify the account

module load python/anaconda-2019.03
module load intelmpi/2019.up7+intel-19.1.1
module load cuda

mpirun -n 2 python3 test.py > test.out