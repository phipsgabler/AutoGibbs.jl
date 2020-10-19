#!/bin/sh
#SBATCH -D /clusterFS/home/user/phg/git/AutoGibbs.jl
#SBATCH --export=HOME=/clusterFS/home/user/phg,TERM,JULIA_DEPOT_PATH=/clusterFS/home/user/phg/julia_depot

julia="julia_1.3.1.afs --project"
homepath="/clusterFS/home/user/${USER}"
codepath="${homepath}/git/AutoGibbs.jl"
runpath="${codepath}/test/experiments.jl"
resultspath="${codepath}/results"

cd ${codepath}

model="gmm"
${julia} ${runpath} ${model} ${resultspath}

# Running an interactive session:
# srun -J "test" -p labor  -l test.sh
# srun -J "autogibbs" -p simulation12 slurm/run.sh
