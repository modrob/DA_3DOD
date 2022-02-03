#INFERNCE_SAMPLE_TEXTFILE = $1
#SOURCE_MODEL = $2
#TARGET_DATASET = $3 

##Script to move inference samples (pseudo-labels) to multi-data for training/finetuning

echo $1
echo $2
echo $3

cd /scratch/ws/1/s1510289-KITTI/3D_adapt_auto_driving/pointrcnn/output/evaluation/sn_${2}_${3}_subsample_200_train_it_3/eval/epoch_190/train/final_result/data
mkdir /scratch/ws/1/s1510289-KITTI/3D_adapt_auto_driving/pointrcnn/multi_data/${3}/KITTI/object/training/pred_sn_${2}_it_3
cat ${1} | xargs -I % cp %.txt /scratch/ws/1/s1510289-KITTI/3D_adapt_auto_driving/pointrcnn/multi_data/${3}/KITTI/object/training/pred_sn_${2}_it_3
cp ${1} /scratch/ws/1/s1510289-KITTI/3D_adapt_auto_driving/pointrcnn/multi_data/${3}/KITTI/ImageSets/sn_${2}_it_3_${1}


#Sample Execution for source-model kitti and target data waymo:
#mv_inf_sample_to_multidata.sh output.txt kitti waymo

