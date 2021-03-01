#! /bin/sh
#SBATCH --job-name=awesome
#SBATCH --output=awesome.out
#SBATCH --error=awesome.err
#SBATCH --partition=studentbatch
#SBATCH --job-name hello-world
#SBATCH --time=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
echo hi > /home/joberant/nlp_fall_2021/shlomotannor/transform-and-tell/abc
