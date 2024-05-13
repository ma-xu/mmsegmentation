#!/bin/bash
#SBATCH -N 1
#SBATCH -p ai-jumpstart
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512Gb
#SBATCH --time=1-23:59:00
#SBATCH --output=%j-livis-c-2.log


cd /dev/shm/
cp /work/smile/xuma/data/ade/ADEChallengeData2016.zip .
unzip ADEChallengeData2016.zip
rm ADEChallengeData2016.zip
cd /scratch/ma.xu1/mmsegmentation
ln -s /dev/shm/ADEChallengeData2016 data/ade
source activate /work/smile/xuma/conda/envs/mmseg

sh tools/dist_train.sh configs/livis/segformer_livis_c_run2.py 8 --work-dir work_dirs/livis_c_2
