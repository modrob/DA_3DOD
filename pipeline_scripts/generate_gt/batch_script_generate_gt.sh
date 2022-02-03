#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --job-name=generate_gt_waymo
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 12
#SBATCH --mem-per-cpu=2541
#SBATCH --mail-user=moritz.drobnitzky@web.de
#SBATCH --mail-type=ALL
#SBATCH --output=gen_gt_rescaled_nusc_waymo-%j.out
#SBATCH --error=gen_gt_rescaled_nusc_waymo-%j.err

module purge
module modenv/scs5                          # Set up environment, e.g., clean modules environment
module load Python/3.6.6-fosscuda-2018b                 # and load necessary modules
source /scratch/ws/1/s1510289-KITTI/venv_3DOD/bin/activate

cd /scratch/ws/1/s1510289-KITTI/3D_adapt_auto_driving/pointrcnn/tools/
srun python generate_gt_database.py --root /scratch/ws/1/s1510289-KITTI/3D_adapt_auto_driving/pointrcnn/multi_data/nusc_to_waymo --save_dir ./gt_database/rescaled/nusc_waymo
