#!/bin/bash

#SBATCH --job-name=17
#SBATCH --output=/ukp-storage-1/ychen/syn_server/Reproducibility/WMT17_Mover/0.txt
#SBATCH --mail-user=chenyr1996@hotmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1


export TOKENIZERS_PARALLELISM=true
export MPLCONFIGDIR='/ukp-storage-1/ychen/syn_server/.config/matplotlib'

cd /ukp-storage-1/ychen/syn_server/Reproducibility/WMT17_Mover

pwd
#source /ukp-storage-1/ychen/py36/bin/activate
source /ukp-storage-1/ychen/anaconda3/etc/profile.d/conda.sh
conda activate dl
module load cuda/11.1

python reproduce_17.py  --metric baryscore --model bert-base-uncased
python reproduce_17.py   --metric bertscore --model bert-base-uncased
python reproduce_17.py  --metric moverscore --model original
python reproduce_17.py  --metric moverscore --model bert-base-uncased
