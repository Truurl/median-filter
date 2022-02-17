#!/bin/bash -l
#SBATCH -J median_filter_job_omp
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:10:00
#SBATCH -p plgrid-testing
#SBATCH --output=omp_median_filter.out
cd $HOME/median-filter/omp
module -q add plgrid/libs/opencv
module -q add plgrid/tools/intel/2021.3.0
make
./median_filter_omp ../lena_noise1.png
./median_filter_omp ../lena_noise2.png
./median_filter_omp ../lena_noise3.png
