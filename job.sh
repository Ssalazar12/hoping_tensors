#!/usr/bin/bash

# name of queue job
#$ -N test

# write errors and logfiles in folder log/
# folder MUST EXIST!
#$ -e logs/ # give absolut path to the desired location
#$ -o logs/
#$ -l h_vmem=30G
# start program, if excecutable of course just call it
module load conda
source activate qpc_venv
python /home/user/santiago.salazar-jaramillo/hoping_tensors/scripts/qpc_qubit_exact_diag.py
