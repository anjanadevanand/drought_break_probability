#!/bin/bash

#PBS -P w97
#PBS -q normalbw
#PBS -l walltime=4:00:00
#PBS -l jobfs=100gb
#PBS -l mem=32GB
#PBS -l ncpus=8
#PBS -l storage=gdata/w97+gdata/hh5
#PBS -m abe
#PBS -M anjana.devanand@unsw.edu.au
#PBS -N glm_test

#####PBS -j oe
#####PBS -l wd

module use /g/data/hh5/public/modules
module load conda/analysis3

export iWeek=6
python ./fit_logiReg_gridded_varyThresh.py