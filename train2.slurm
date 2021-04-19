#! /bin/sh
#SBATCH --mail-user=maorkehati@mail.tau.ac.il
#SBATCH --mail-type=ALL
#SBATCH --output=tr2a2_%j.out
#SBATCH --error=tr2a2_%j.err
#SBATCH --partition=studentkillable
#SBATCH --job-name tr2a2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
/specific/netapp5/joberant/nlp_fall_2021/shlomotannor/anaconda3/envs/tell/bin/python trainBM2.py Tarch1_1024_512_1 arch1

