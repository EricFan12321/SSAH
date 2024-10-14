#!/bin/bash -l
#SBATCH --partition=gpu-a100
#SBATCH --job-name=experiment1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1  # Request 1 GPU
#SBATCH --mem=128G
#SBATCH --qos=short
#SBATCH --output=result.out
#SBATCH --error=error.log
#SBATCH --mail-type=END,FAIL # Send email on job END and FAIL
#SBATCH --mail-user=liming.fan333@gmail.com

# Navigate to your project directory
cd /home/user/eric123/experiment1/experiment1.1/SSAH-adversarial-attack

# Load Miniconda
export PATH=~/miniconda3/bin:$PATH

# Initialize Conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate py36

# Check GPU availability
echo "Checking GPU availability with nvidia-smi:"
nvidia-smi

# Run your Python script
srun python main-lpips.py \
    --classifier='resnet20' \
    --dataset='cifar10' \
    --bs=5000 \
    --max-epoch=1 \
    --wavelet='haar' \
    --num-iteration=150 \
    --learning-rate=0.001 \
    --m=0.2 \
    --alpha=1 \
    --lambda-lf=0.1 \
    --seed=8 \
    --workers=32 \
    --test-fid 
# Deactivate the conda environment
conda deactivate