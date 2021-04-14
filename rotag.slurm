#! /bin/sh
#SBATCH --output=rotab_%j.out
#SBATCH --error=rotab_%j.err
#SBATCH --partition=studentbatch
#SBATCH --job-name rotab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
/specific/netapp5/joberant/nlp_fall_2021/shlomotannor/anaconda3/envs/tell/bin/python robertadb.py agentr

