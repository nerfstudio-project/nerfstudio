#!/usr/bin/bash

#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:A100m40:1
#SBATCH --account=ie-idi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32000
#SBATCH --job-name="colmap-exp.out"
#SBATCH --output=colmap-exp.out
#SBATCH --mail-user=olesto@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}

cd ${WORKDIR}
echo "we are running from this directory: $WORKDIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "The job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

# Purge modules
module purge
ml FFmpeg/5.0.1-GCCcore-11.3.0

# Deactivate any spill-over environment from the login node
conda deactivate &>/dev/null

conda info
conda activate nerfstudio 
conda info

# Set env-variable to enable correct colmap call
SLURM_JOB=True
DATA_PATH="data/videos/ohma_electra"

# Run training
./nerf_dataset_exp.py --model nerfacto --data-source video --input-data-dir $DATA_PATH/ohma_electra.MOV --output-dir $DATA_PATH --percent-frames 0.2,0.25,0.33,0.5,1.0 
