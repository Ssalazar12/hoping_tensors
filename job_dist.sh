!/usr/bin/bash

# reserve memory for each job, default 1GB
#$ -l h_vmem=20G
# reserve the cores
#$ -pe smp 8

# name of queue job
#$ -N 'distributed'

# inform via email
#$ -m eba # information when start, error or aborted, and completed 
#$ -M santiago.salazar-jaramillo@uni-konstanz.de

# write errors and logfiles in folder log/
# folder MUST EXIST!1d
#$ -e logs/ # give absolut path to the desired location 
#$ -o logs/

# activate Julia threading
export JULIA_NUM_THREADS=8
module load julia 
# start program
julia /home/user/santiago.salazar-jaramillo/hoping_tensors/scripts/MPS_run.jl
