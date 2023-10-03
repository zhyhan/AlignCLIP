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

# #export CUDA_VISIBLE_DEVICES=0
python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data Caltech101\
       --seed 0\
       --epochs 1\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/Caltech101
python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data Caltech101\
       --seed 1\
       --epochs 1\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/Caltech101
python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data Caltech101\
       --seed 2\
       --epochs 1\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/Caltech101

python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data OxfordPets\
       --seed 0\
       --epochs 1\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/OxfordPets
python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data OxfordPets\
       --seed 1\
       --epochs 1\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/OxfordPets
python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data OxfordPets\
       --seed 2\
       --epochs 1\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/OxfordPets

python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data StanfordCars\
       --seed 0\
       --epochs 5\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/StanfordCars
python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data StanfordCars\
       --seed 1\
       --epochs 5\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/StanfordCars
python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data StanfordCars\
       --seed 2\
       --epochs 5\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/StanfordCars

python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data OxfordFlowers\
       --seed 0\
       --epochs 1\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/OxfordFlowers
python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data OxfordFlowers\
       --seed 1\
       --epochs 1\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/OxfordFlowers
python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data OxfordFlowers\
       --seed 2\
       --epochs 1\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/OxfordFlowers

python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data Food101\
       --seed 0\
       --epochs 1\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/Food101
python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data Food101\
       --seed 1\
       --epochs 1\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/Food101
python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data Food101\
       --seed 2\
       --epochs 1\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/Food101

python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data FGVCAircraft\
       --seed 0\
       --epochs 10\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/FGVCAircraft
python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data FGVCAircraft\
       --seed 1\
       --epochs 10\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/FGVCAircraft
python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data FGVCAircraft\
       --seed 2\
       --epochs 10\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/FGVCAircraft

python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data SUN397\
       --seed 0\
       --epochs 5\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/SUN397
python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data SUN397\
       --seed 1\
       --epochs 5\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/SUN397
python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data SUN397\
       --seed 2\
       --epochs 5\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/SUN397

python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data DescribableTextures\
       --seed 0\
       --epochs 1\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/DescribableTextures
python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data DescribableTextures\
       --seed 1\
       --epochs 1\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/DescribableTextures
python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data DescribableTextures\
       --seed 2\
       --epochs 1\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/DescribableTextures

python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data EuroSAT\
       --seed 0\
       --epochs 1\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/EuroSAT
python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data EuroSAT\
       --seed 1\
       --epochs 1\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/EuroSAT
python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data EuroSAT\
       --seed 2\
       --epochs 1\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/EuroSAT


python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data UCF101\
       --seed 0\
       --epochs 2\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/UCF101
python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data UCF101\
       --seed 1\
       --epochs 2\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/UCF101
python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data UCF101\
       --seed 2\
       --epochs 2\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/UCF101

python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data ImageNet\
       --seed 0\
       --epochs 10\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/ImageNet
python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data ImageNet\
       --seed 1\
       --epochs 10\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/ImageNet
python -m main\
       /l/users/zhongyi.han/dataset/CoOp/\
       --data ImageNet\
       --seed 2\
       --epochs 10\
       --task open_class\
       --beta 0.1\
       --log exps/open_class/ImageNet