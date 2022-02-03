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
#SBATCH --output=gen_gt_nusc_st_sn_kitti_sampling_it_3-%j.out
#SBATCH --error=gen_gt_nusc_st_sn_kitti_sampling_it_3-%j.err

module purge
module modenv/scs5                          # Set up environment, e.g., clean modules environment
module load Python/3.6.6-fosscuda-2018b                 # and load necessary modules
source /scratch/ws/1/s1510289-KITTI/venv_3DOD/bin/activate

cd /scratch/ws/1/s1510289-KITTI/3D_adapt_auto_driving/pointrcnn/tools/
srun python generate_gt_database.py --root /scratch/ws/1/s1510289-KITTI/3D_adapt_auto_driving/pointrcnn/multi_data/kitti --save_dir ./gt_database/sn_self_training/it_3/kitti --self_training pred_sn_nusc_it_3 --self_training_textfile sn_nusc_it_3_output.txt
