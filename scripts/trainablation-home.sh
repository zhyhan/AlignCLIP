#!/bin/bash
#SBATCH --job-name=ablation         # Job name
#SBATCH --output=./logs/output.%A_%a.txt   # Standard output and error log
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=40G                   # Total RAM to be used
#SBATCH --cpus-per-task=64          # Number of CPU cores
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH -p gpu                      # Use the gpu partition
#SBATCH --time=12:00:00             # Specify the time needed for your experiment
#SBATCH --qos=gpu-8                 # To enable the use of up to 8 GPUs

#export CUDA_VISIBLE_DEVICES=2
python -m main\
       /home/zhongyi.han/dataset/domainbed/\
       --data OfficeHome\
       --seed 0\
       --targets 0\
       --log exps/ablation/OfficeHome\
       --aloss 0\
       --lr 1e-5\
       --layer 1\

# python -m main\
#        /home/zhongyi.han/dataset/domainbed/\
#        --data OfficeHome\
#        --seed 0\
#        --targets 1\
#        --log exps/ablation/OfficeHome\
#        --aloss 0\
#        --lr 1e-5\
#        --layer 1\

# python -m main\
#        /home/zhongyi.han/dataset/domainbed/\
#        --data OfficeHome\
#        --seed 0\
#        --targets 2\
#        --log exps/ablation/OfficeHome\
#        --aloss 0\
#        --lr 1e-5\
#        --layer 1\

# python -m main\
#        /home/zhongyi.han/dataset/domainbed/\
#        --data OfficeHome\
#        --seed 0\
#        --targets 3\
#        --log exps/ablation/OfficeHome\
#        --aloss 0\
#        --lr 1e-5\
#        --layer 1\

# python -m main\
#        /home/zhongyi.han/dataset/domainbed/\
#        --data OfficeHome\
#        --seed 0\
#        --targets 0\
#        --log exps/ablation/OfficeHome\
#        --aloss 1\
#        --layer 3\

# python -m main\
#        /home/zhongyi.han/dataset/domainbed/\
#        --data OfficeHome\
#        --seed 0\
#        --targets 1\
#        --log exps/ablation/OfficeHome\
#        --aloss 1\
#        --layer 3\

# python -m main\
#        /home/zhongyi.han/dataset/domainbed/\
#        --data OfficeHome\
#        --seed 0\
#        --targets 2\
#        --log exps/ablation/OfficeHome\
#        --aloss 1\
#        --layer 3\

# python -m main\
#        /home/zhongyi.han/dataset/domainbed/\
#        --data OfficeHome\
#        --seed 0\
#        --targets 3\
#        --log exps/ablation/OfficeHome\
#        --aloss 1\
#        --layer 3\

# python -m main\
#        /home/zhongyi.han/dataset/domainbed/\
#        --data OfficeHome\
#        --seed 0\
#        --targets 0\
#        --log exps/ablation/OfficeHome\
#        --aloss 1\
#        --layer 5\

# python -m main\
#        /home/zhongyi.han/dataset/domainbed/\
#        --data OfficeHome\
#        --seed 0\
#        --targets 1\
#        --log exps/ablation/OfficeHome\
#        --aloss 1\
#        --layer 5\

# python -m main\
#        /home/zhongyi.han/dataset/domainbed/\
#        --data OfficeHome\
#        --seed 0\
#        --targets 2\
#        --log exps/ablation/OfficeHome\
#        --aloss 1\
#        --layer 5\

# python -m main\
#        /home/zhongyi.han/dataset/domainbed/\
#        --data OfficeHome\
#        --seed 0\
#        --targets 3\
#        --log exps/ablation/OfficeHome\
#        --aloss 1\
#        --layer 5\

# python -m main\
#        /home/zhongyi.han/dataset/domainbed/\
#        --data OfficeHome\
#        --seed 0\
#        --targets 0\
#        --log exps/ablation/OfficeHome\
#        --aloss 0.8\
#        --layer 2\

# python -m main\
#        /home/zhongyi.han/dataset/domainbed/\
#        --data OfficeHome\
#        --seed 0\
#        --targets 1\
#        --log exps/ablation/OfficeHome\
#        --aloss 0.8\
#        --layer 2\

# python -m main\
#        /home/zhongyi.han/dataset/domainbed/\
#        --data OfficeHome\
#        --seed 0\
#        --targets 2\
#        --log exps/ablation/OfficeHome\
#        --aloss 0.8\
#        --layer 2\

# python -m main\
#        /home/zhongyi.han/dataset/domainbed/\
#        --data OfficeHome\
#        --seed 0\
#        --targets 3\
#        --log exps/ablation/OfficeHome\
#        --aloss 0.8\
#        --layer 2\