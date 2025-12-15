#!/bin/bash
#SBATCH --job-name=mpi_mat_batch
#SBATCH --partition=batch          # <== NOT testing
#SBATCH --nodes=2                  # 2 nodes
#SBATCH --ntasks-per-node=2        # 2 MPI ranks per node (total 4)
#SBATCH --cpus-per-task=2          # 2 OpenMP threads per rank
#SBATCH --time=00:01:00
#SBATCH --output=OUT_%j.out
#SBATCH --error=ERR_%j.err

module purge
module load gcc/9.2.0 openmpi/3.1.3

export MATIO_ROOT=/mnt/apps/aristotle/site/linux-centos7-x86_64/gcc-9.4.0/matio-1.5.26-lrqns6gbz3i2zgqhoaml3svuz7jyg3qp
export HDF5_ROOT=/mnt/apps/aristotle/site/linux-centos7-x86_64/gcc-9.4.0/hdf5-1.14.3-swov2l62sfp7i3k76j2wqxtsqgbwyu53

echo "== Compiling =="
mpic++ -O2 -fopenmp mpi_mat_prog.cpp helper_functions.cpp \
  -I$MATIO_ROOT/include \
  -I$HDF5_ROOT/include \
  -L$MATIO_ROOT/lib \
  -L$HDF5_ROOT/lib \
  -o mpi_mat_prog \
  -lmatio -lhdf5

echo "== Running =="
export LD_LIBRARY_PATH=$MATIO_ROOT/lib:$HDF5_ROOT/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

MAT_FILE=roadNet-CA.mat

echo "Running on host: $(hostname)"
echo "NODES=$SLURM_JOB_NUM_NODES, NTASKS=$SLURM_NTASKS, OMP_THREADS=$OMP_NUM_THREADS"

mpiexec -np ${SLURM_NTASKS} ./mpi_mat_prog "$MAT_FILE" Problem A

echo "== Done =="
