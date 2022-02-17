#!/bin/bash -l
#SBATCH -J median_filter_cuba_job
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --partition=plgrid-gpu
#SBATCH -A plgpiask2021
#SBATCH --output=cuda_median_filter.out
cd $HOME/median-filter/cuda
module -q add plgrid/libs/opencv
module -q add plgrid/apps/cuda
make
./median_filter_cuda ../lena_noise1.png
./median_filter_cuda ../lena_noise2.png
./median_filter_cuda ../lena_noise3.png
