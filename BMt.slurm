#! /bin/sh
#SBATCH --output=BMt_%j.out
#SBATCH --error=BMt_%j.err
#SBATCH --partition=studentbatch
#SBATCH --job-name newscaptioning1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
/specific/netapp5/joberant/nlp_fall_2021/shlomotannor/anaconda3/envs/tell/bin/python trainBM.py
