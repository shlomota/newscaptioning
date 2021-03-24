#! /bin/sh
#SBATCH --job-name=pocc
#SBATCH --output=pocc.out
#SBATCH --error=pocc.err
#SBATCH --partition=studentbatch
#SBATCH --job-name newscaptioningg1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
/specific/netapp5/joberant/nlp_fall_2021/shlomotannor/anaconda3/envs/tell/bin/python poc_scriptt.py

