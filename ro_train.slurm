#! /bin/sh
#SBATCH --output=rotrain_%j.out
#SBATCH --error=rotrain_%j.err
#SBATCH --partition=studentbatch
#SBATCH --job-name ro_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
/specific/netapp5/joberant/nlp_fall_2021/shlomotannor/anaconda3/envs/tell/bin/python robertadb.py

