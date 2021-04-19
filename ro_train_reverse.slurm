#! /bin/sh
#SBATCH --output=rotrainr_%j.out
#SBATCH --error=rotrainr_%j.err
#SBATCH --partition=studentbatch
#SBATCH --job-name ro_tr_r
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
/specific/netapp5/joberant/nlp_fall_2021/shlomotannor/anaconda3/envs/tell/bin/python robertadb.py train r

