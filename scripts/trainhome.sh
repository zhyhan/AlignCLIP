#!/bin/bash
#SBATCH --job-name=CLIPood             # Job name
#SBATCH --output=./logs/output.%A_%a.txt   # Standard output and error log
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=40G                   # Total RAM to be used
#SBATCH --cpus-per-task=64          # Number of CPU cores
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH -p gpu                      # Use the gpu partition
#SBATCH --time=12:00:00             # Specify the time needed for your experiment
#SBATCH --qos=gpu-8                 # To enable the use of up to 8 GPUs

#export CUDA_VISIBLE_DEVICES=0
python -m main\
       /home/zhongyi.han/dataset/domainbed/\
       --data OfficeHome\
       --lr 1e-5\
       --seed 0\
       --targets 0\
       --task domain_shift\
       --log exps/domain_shift/OfficeHome
#export CUDA_VISIBLE_DEVICES=0
python -m main\
       /home/zhongyi.han/dataset/domainbed/\
       --data OfficeHome\
       --lr 1e-5\
       --seed 0\
       --targets 1\
       --task domain_shift\
       --log exps/domain_shift/OfficeHome
#export CUDA_VISIBLE_DEVICES=0
python -m main\
       /home/zhongyi.han/dataset/domainbed/\
       --data OfficeHome\
       --lr 1e-5\
       --seed 0\
       --targets 2\
       --task domain_shift\
       --log exps/domain_shift/OfficeHome
#export CUDA_VISIBLE_DEVICES=0
python -m main\
       /home/zhongyi.han/dataset/domainbed/\
       --data OfficeHome\
       --lr 1e-5\
       --seed 0\
       --target 3\
       --task domain_shift\
       --log exps/domain_shift/OfficeHome

#export CUDA_VISIBLE_DEVICES=0
python -m main\
       /home/zhongyi.han/dataset/domainbed/\
       --data OfficeHome\
       --lr 1e-5\
       --seed 1\
       --targets 0\
       --task domain_shift\
       --log exps/domain_shift/OfficeHome
#export CUDA_VISIBLE_DEVICES=0
python -m main\
       /home/zhongyi.han/dataset/domainbed/\
       --data OfficeHome\
       --lr 1e-5\
       --seed 1\
       --targets 1\
       --task domain_shift\
       --log exps/domain_shift/OfficeHome
#export CUDA_VISIBLE_DEVICES=0
python -m main\
       /home/zhongyi.han/dataset/domainbed/\
       --data OfficeHome\
       --lr 1e-5\
       --seed 1\
       --targets 2\
       --task domain_shift\
       --log exps/domain_shift/OfficeHome
#export CUDA_VISIBLE_DEVICES=0
python -m main\
       /home/zhongyi.han/dataset/domainbed/\
       --data OfficeHome\
       --lr 1e-5\
       --seed 1\
       --target 3\
       --task domain_shift\
       --log exps/domain_shift/OfficeHome

#export CUDA_VISIBLE_DEVICES=0
python -m main\
       /home/zhongyi.han/dataset/domainbed/\
       --data OfficeHome\
       --lr 1e-5\
       --seed 2\
       --targets 0\
       --task domain_shift\
       --log exps/domain_shift/OfficeHome
#export CUDA_VISIBLE_DEVICES=0
python -m main\
       /home/zhongyi.han/dataset/domainbed/\
       --data OfficeHome\
       --lr 1e-5\
       --seed 2\
       --targets 1\
       --task domain_shift\
       --log exps/domain_shift/OfficeHome
#export CUDA_VISIBLE_DEVICES=0
python -m main\
       /home/zhongyi.han/dataset/domainbed/\
       --data OfficeHome\
       --lr 1e-5\
       --seed 2\
       --targets 2\
       --task domain_shift\
       --log exps/domain_shift/OfficeHome
#export CUDA_VISIBLE_DEVICES=0
python -m main\
       /home/zhongyi.han/dataset/domainbed/\
       --data OfficeHome\
       --lr 1e-5\
       --seed 2\
       --target 3\
       --task domain_shift\
       --log exps/domain_shift/OfficeHome