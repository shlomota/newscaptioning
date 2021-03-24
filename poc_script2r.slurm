#! /bin/sh
#SBATCH --output=poc2r.out
#SBATCH --error=poc2r.err
#SBATCH --partition=studentbatch
#SBATCH --job-name newscaptioning1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
/specific/netapp5/joberant/nlp_fall_2021/shlomotannor/anaconda3/envs/tell/bin/python poc_script2.py

