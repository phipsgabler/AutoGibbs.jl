#!/bin/sh
#SBATCH 

julia="julia_1.3.1.afs --project"
homepath="/clusterFS/home/user/${USER}"
codepath="${homepath}/git/AutoGibbs.jl"
runpath="${codepath}/test/experiments.jl"
resultspath="${codepath}/results"

cd ${codepath}

model="gmm"
${julia} ${runpath} ${model} ${resultspath}

# Running an interactive session:
# srun -J "test julia" -p labor -D /clusterFS/home/user/phg --export=HOME=/clusterFS/home/user/phg,TERM,JULIA_DEPOT_PATH=/clusterFS/home/user/phg/julia_depot -l test.sh
# srun -p simulation12 --mem 20G --pty /bin/bash -l
# --cpus-per-task 1 --time 48:00:00 -D /clusterFS/home/user/${USER} --export=HOME=/clusterFS/home/user/${USER},TERM,JULIA_DEPOT_PATH=/clusterFS/home/user/phg/julia_depot
