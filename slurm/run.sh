#!/bin/bash
#SBATCH --nodes 1 # request one node
#SBATCH --mem-per-cpu 5G
#SBATCH --cpus-per-task 1
#SBATCH --time 12:00:00
#SBATCH --exclusive
#SBATCH --array=0-1
#SBATCH --export HOME=/clusterFS/home/user/phg,TERM,JULIA_DEPOT_PATH=/clusterFS/home/user/phg/julia_depot
#SBATCH --mail-type END,FAIL

julia="/afs/spsc.tugraz.at/opt/julia/1.3.1/bin/julia --project"
homepath="/clusterFS/home/user/${USER}"
codepath="${homepath}/git/AutoGibbs.jl"
runpath="${codepath}/test/experiments.jl"
resultspath="${codepath}/results"
outfile="${resultspath}/out.txt"
errfile="${resultspath}/err.txt"

run_slurm="srun -n1 -o ${outfile} -e ${errfile}"

cd ${codepath}

MODELS=(gmm hmm imm)
MAX_SIZES=(100 100 10)
${run_slurm} ${julia} ${runpath} ${MODELS[$SLURM_ARRAY_TASK_ID]} ${MAX_SIZES[$SLURM_ARRAY_TASK_ID]} ${resultspath}

# test julia
# ${run_slurm} -l ${outfile} -e ${errfile} ${julia} -e "sleep(20); println(\"${MODELS[$SLURM_ARRAY_TASK_ID]}\", ${SLURM_ARRAY_TASK_ID})"

# Running an interactive session:
# srun -J "test_julia" -p labor -D /clusterFS/home/user/phg/git/AutoGibbs.jl /bin/bash

# Batch command
# sbatch -J "autogibbs" -p labor --exclude=dsplab[01-12],tonlabor[1-9],tonstudio1 -D /clusterFS/home/user/phg/git/AutoGibbs.jl slurm/run.sh
