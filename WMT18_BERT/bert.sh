#!/bin/bash

#SBATCH --job-name=18baryb
#SBATCH --output=/ukp-storage-1/ychen/syn_server/Reproducibility/WMT18_BERT/bary.txt
#SBATCH --mail-user=chenyr1996@hotmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --nodelist=melvin

export TOKENIZERS_PARALLELISM=true
export MPLCONFIGDIR='/ukp-storage-1/ychen/syn_server/.config/matplotlib'

cd /ukp-storage-1/ychen/syn_server/Reproducibility/WMT18_BERT

pwd
#source /ukp-storage-1/ychen/py36/bin/activate
source /ukp-storage-1/ychen/anaconda3/etc/profile.d/conda.sh
conda activate dl
module load cuda/11.1

python reproduce_18.py  --metric baryscore
#python reproduce_18.py  --metric moverscore
#python reproduce_18.py   --metric bertscore
