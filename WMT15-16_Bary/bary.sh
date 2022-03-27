#!/bin/bash

#SBATCH --job-name=15160
#SBATCH --output=/ukp-storage-1/ychen/syn_server/Reproducibility/WMT15-16_Bary/bary1.txt
#SBATCH --mail-user=chenyr1996@hotmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1


export TOKENIZERS_PARALLELISM=true
export MPLCONFIGDIR='/ukp-storage-1/ychen/syn_server/.config/matplotlib'

cd /ukp-storage-1/ychen/syn_server/Reproducibility/WMT15-16_Bary

pwd
#source /ukp-storage-1/ychen/py36/bin/activate
source /ukp-storage-1/ychen/anaconda3/etc/profile.d/conda.sh
conda activate dl
module load cuda/11.1

python reproduce_1516.py  --metric baryscore --model bert-base-uncased --dataset 15
python reproduce_1516.py  --metric baryscore --model original --dataset 15
#python reproduce_1516.py   --metric bertscore --model bert-base-uncased --dataset 15
#python reproduce_1516.py  --metric moverscore --model original --dataset 15
#python reproduce_1516.py  --metric moverscore --model bert-base-uncased --dataset 15

python reproduce_1516.py  --metric baryscore --model bert-base-uncased --dataset 16
python reproduce_1516.py  --metric baryscore --model original --dataset 16
#python reproduce_1516.py   --metric bertscore --model bert-base-uncased --dataset 16
#python reproduce_1516.py  --metric moverscore --model original --dataset 16
#python reproduce_1516.py  --metric moverscore --model bert-base-uncased --dataset 16
