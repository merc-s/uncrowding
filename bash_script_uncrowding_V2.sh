#!/bin/bash
#SBATCH --chdir /scratch/simos
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 128G
#SBATCH --time 24:00:00
echo STARTING AT `date`
python /home/simos/uncrowding/uncrowding_pipeline_uncrowding_V2.py
echo FINISHED AT `date`