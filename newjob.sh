#!/bin/sh
# =============================================================================
#
# This file creates a new job with a PBS template.
# It looks inside "./jobs" subdir for the last job and increments it
# call it like this:
#
# $ ./newjob.sh <script_name> [hours epochs split mem] 
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
	walltime=24:00:00
fi

# SET EPOCHS
if [[ $3 ]]; then
	EPOCHS=$3
else
	EPOCHS=10
fi

# SET S_TRAIN
if [[ $4 ]]; then
	S_TRAIN=$4
else
	S_TRAIN=0
fi

# SET mem per cpu
if [[ $5 && $5 -gt 0 ]]; then
	MEM=$5
else
	MEM=4000
fi


# SET MESSAGING WHEN JOB BEGINS/ENDS
EMAIL=camarahalac.1@osu.edu
recalls=ALL

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
modelpath=${path}/saved_model
DATADIR=~/nsynth

job_prefix=0
i=0

if ! test -f ${job_path}/0-*.sh; then 
	job_prefix=0
else
	# GET LATEST PREFIX AND INCREMENT IT
	l=$(ls ${job_path} | grep -v readme | sort -n)
	for i in ${l[@]}
	do
		i=$i
	done
	job_prefix=$(basename $i "-train.sh" | cut -f1 -d- )
	job_prefix=$((job_prefix+=1))
fi

env=/bin/bash # job environment
job_name=${job_prefix}${job_suffix}
job_output=${job_name}
venv=${path}/../.venvs/${which_venv}/bin/activate
script_path=${path}/${script}

function printSBATCH() {
	local h=$1
	printf "%s %s\n" "#SBATCH" "${h}"
}

# SET CLUSTER PATH NAME
name=PAS1309
j="${job_path}/${job_name}.sh"
printf "%s\n" "#!/bin/bash" > $j
printSBATCH "--time=${walltime}" >> $j
printSBATCH "--nodes=${nodes} --ntasks-per-node=${ppn} --gpus-per-node=${gpus}  --gpu_cmode=shared" >> $j
printSBATCH "--account=${name} " >> $j
printSBATCH "--job-name=${job_output}" >> $j
printSBATCH "--mem-per-cpu=${MEM}" >> $j
printSBATCH "--mail-type=${recalls}" >> $j 
printSBATCH "--mail-user=${EMAIL}" >> $j 
printSBATCH "--error=${job_logs}/${job_name}-e.txt" >> $j
printSBATCH "--output=${job_logs}/${job_name}-o.txt" >> $j
printf "#" >> $j
printf "=%.0s" {1..80} >> $j
echo >> $j
echo "# job created on" $(date) >> $j
echo "# with these arguments:" >> $j
echo "# $@" >> $j
echo "#" >> $j
printf "%s %s\n"    "source" "${venv}" >> $j
printf "%s %s %s\n" "module" "load" "${cuda}" >> $j
printf "%s %s %s %s %s %s %s\n" "python" "${script_path}" "$job_prefix" "$modelpath" "$EPOCHS" "$S_TRAIN" "$DATADIR" >> $j
echo "Finished making job: $j"
echo "with these arguments: $@"