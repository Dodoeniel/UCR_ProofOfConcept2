#!/bin/bash

#SBATCH -p gpu
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --constraint=OS7
#SBATCH --time 24:00:00
#SBATCH --mail-type ALL
#SBATCH --mail-user daniel.schoepflin@tuhh.de
#SBATCH --nodelist d050
#SBATCH --exclusive
#SBATCH --gres gpu:1

# load cuda
. /etc/profile.d/module.sh
module load cuda/9.0

MYWORKDIR=/work/dyn/ctm9918/UCR_ProofOfConcept/$SLURM_JOBID
mkdir $MYWORKDIR

cd $MYWORKDIR
python3 /work/dyn/ctm9918/UCR_ProofOfConcept/scratch.py

# rm -rf $MYWORKDIR

exit
