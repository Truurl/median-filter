#!/bin/bash -l
#SBATCH -J median_filter_job
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH -p plgrid-testing
#SBATCH --output=median_filter.out
cd $HOME/median-filter/base
module -q add plgrid/libs/opencv
module -q add plgrid/tools/gcc/8.2.0
make
./median_filter_normal 3 3 ../lena_noise3.png
