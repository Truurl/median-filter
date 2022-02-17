#!/bin/bash -l
#SBATCH -J mpi_median_filter_job
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:10:00
#SBATCH -p plgrid-testing
#SBATCH --output=mpi_median_filter.out
cd $HOME/median-filter/mpi
module -q add plgrid/libs/opencv
module -q add plgrid/tools/openmpi
make
mpiexec -np 2 ./median_filter_mpi
mpiexec -np 4 ./median_filter_mpi
mpiexec -np 8 ./median_filter_mpi
