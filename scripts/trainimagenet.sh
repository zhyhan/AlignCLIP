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


#export CUDA_VISIBLE_DEVICES=3
# python -m main\
#        /l/users/zhongyi.han/dataset/CoOp\
#        --data ImageNet\
#        --seed 0\
#        --targets 1\
#        --task open_class\
#        --log exps/domain_shift/ImageNet

export CUDA_VISIBLE_DEVICES=3
python -m main\
       /l/users/zhongyi.han/dataset/CoOp\
       --data ImageNetSketch\
       --seed 0\
       --targets 1\
       --epochs 0\
       --task open_class\
       --log exps/domain_shift/ImageNetSketch

export CUDA_VISIBLE_DEVICES=3
python -m main\
       /l/users/zhongyi.han/dataset/CoOp\
       --data ImageNetV2\
       --seed 0\
       --targets 1\
       --epochs 0\
       --task open_class\
       --log exps/domain_shift/ImageNetV2

export CUDA_VISIBLE_DEVICES=3
python -m main\
       /l/users/zhongyi.han/dataset/CoOp\
       --data ImageNetR\
       --seed 0\
       --targets 1\
       --epochs 0\
       --task open_class\
       --log exps/domain_shift/ImageNetR

export CUDA_VISIBLE_DEVICES=3
python -m main\
       /l/users/zhongyi.han/dataset/CoOp\
       --data ImageNetA\
       --seed 0\
       --targets 1\
       --epochs 0\
       --task open_class\
       --log exps/domain_shift/ImageNetA