#!/bin/sh
# =============================================================================
#
# This file creates a new job with a PBS template.
# It looks inside "./jobs" subdir for the last job and increments it
#
# =============================================================================

# PYTHON SCRIPT FILENAME GOES HERE
script=kwgan-c.py

# SET WALL TIME 
walltime=24:00:00

# SET MESSAGING WHEN JOB BEGINS/ENDS
recalls=abe

# SET CUDA MODULE TO LOAD
cuda=cuda/10.1.168

# SELECT NODES AND GPUS
nodes=1
ppn=28
gpus=1

# SET VIRTUAL ENVIRONMENT ROOT DIR (LOCATED ON '~/.venvs/...')
which_venv=tf-gpu2.1-3.7

# =============================================================================
# Shouldn't edit what follows unless we change stuff
# =============================================================================

# MATCH REPOSITORY TREE HERE
path=$(pwd)
job_logs=${path}/logs
job_path=${path}/jobs
job_suffix="-train"

# GET LATEST PREFIX AND INCREMENT IT
job_prefix=0
for i in ${job_path}/*.sh
do
	job_prefix=$(basename $i "-train.sh" | cut -f1 -d- )
done
job_prefix=$((job_prefix+=1))

env=/bin/bash # job environment
job_name=${job_prefix}${job_suffix}
job_output=${job_name}
venv=${path}/../.venvs/${which_venv}/bin/activate
script_path=${path}/${script}

function printPBS() {
	local h=$1
	local c=$2
	printf "%s %s %s\n" "#PBS" "${h}" "${c}"
}

# SET CLUSTER PATH NAME
name=PAS1309
j="${job_path}/${job_name}.sh"
printPBS "-l" "walltime=${walltime}" > $j
printPBS "-l" "nodes=${nodes}:ppn=${ppn}:gpus=${gpus}:default" >> $j
printPBS "-A" "${name} " >> $j
printPBS "-N" "${job_output}" >> $j
printPBS "-m" "${recalls}" >> $j
printPBS "-S" "${env}" >> $j
printPBS "-e" "${job_logs}/${job_name}-e.txt" >> $j
printPBS "-o" "${job_logs}/${job_name}-o.txt" >> $j
printf "=%.0s" {1..80} >> $j
echo >> $j
echo "# job created on" $(date) >> $j
printf "%s %s\n"    "source" "${venv}" >> $j
printf "%s %s %s\n" "module" "load" "${cuda}" >> $j
printf "%s %s %s\n" "python" "${script_path}" "$job_prefix" >> $j
echo "Finished making job: $j"