#!/bin/bash
#SBATCH --job-name=SR_acs_v1-6c_Cr_1_Ch_1B_2NH2_Ch_2x2x2_ma5000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000
#SBATCH --export=ALL
#SBATCH --time=4-00:00

module load compilers/gcc/9
module load mpi/openmpi/gcc/3.0.1
module load apps/python3/2020.02
module load apps/lammps/gcc/29oct20

export OMP_NUM_THREADS=1

srun python FL_path_FE.py forward.dat backward.dat force_k.dat 300.0 221537.099969021 > FL_FE.txt

