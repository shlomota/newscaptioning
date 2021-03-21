#! /bin/sh
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --partition=studentbatch
#SBATCH --job-name newscaptioning_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
/specific/netapp5/joberant/nlp_fall_2021/shlomotannor/anaconda3/envs/tell/bin/python trainBM.py

