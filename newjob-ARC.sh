#!/bin/sh
# =============================================================================
#
# This file creates a new job with a SBATCH template.
# It looks inside "./jobs" subdir for the last job and increments it
#
# =============================================================================

# PYTHON SCRIPT FILENAME GOES IN ARGUMENT 1
if [[ $1 ]]; then
	script=$1
else
	echo "Add input script. Exiting."
	exit
fi

# SET WALL TIME
if [[ $2 && $2 -gt 0 ]]; then
	walltime=$2:00:00
else
	echo "Add input walltime. Exiting."
	exit
fi

# SET MESSAGING WHEN JOB BEGINS/ENDS
recalls=abe

# SET CUDA MODULES TO LOAD
conda=anaconda3/2019.03
cuda=gpu/cuda/10.1.243
cudnn=gpu/cudnn/7.6.5__cuda-10.1

# SELECT NODES AND GPUS
nodes=1
# ppn=28
gpus=1

# SET VIRTUAL ENVIRONMENT ROOT DIR (LOCATED ON '~/.venvs/...')
which_venv=tensor-env

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

# PATHs
job_name=${job_prefix}${job_suffix}
job_output=${job_name}
script_path=${path}/${script}

# Print function for writing SBATCH
function printSBATCH() {
	local h=$1
	printf "%s %s\n" "#SBATCH" "${h}"
}

# SET CLUSTER PATH NAME
name=KWAMEH
j="${job_path}/${job_name}.sh"

printf "%s\n" "#!/bin/bash" >$j
printSBATCH "--nodes=${nodes}" >> $j
printSBATCH "--ntasks-per-node=1" >> $j
printSBATCH "--time=${walltime}" >> $j
printSBATCH "--job-name=${name}" >> $j
printSBATCH "--partition=htc" >> $j
printSBATCH "--gres=gpu:${gpus}" >> $j
printSBATCH "--mail-type=ALL" >> $j
printSBATCH "--mail-user=matias.delgadino@maths.ox.ac.uk" >> $j
printSBATCH "--output=${job_logs}/${job_name}.out" >> $j
printSBATCH "--error=${job_logs}/${job_name}.err" >> $j

printf "#" >> $j
printf "=%.0s" {1..80} >> $j
echo >> $j
echo "# job created on" $(date) >> $j

# MODULE LOAD
printf "%s %s %s\n" "module" "load" "${conda}" >> $j
printf "%s %s %s\n" "module" "load" "${cuda}" >> $j
printf "%s %s %s\n" "module" "load" "${cudnn}" >> $j

# ACTIVATE ENVIROMENT
printf "%s %s %s\n"    "source" "activate" "/data/math-gan-pdes/math1656/${which_venv}" >> $j

# RUN SCRIPT
printf "%s %s %s\n" "python" "${script_path}" "$job_prefix" >> $j
echo "Finished making job: $j $walltime"
