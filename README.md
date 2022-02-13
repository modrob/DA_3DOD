# DA_3DOD

## Dependencies
- [Python 3.6.10](https://www.python.org/downloads/)
- [PyTorch(1.0.0)](http://pytorch.org)

further package dependencies are listed in requirements.txt


## Usage

### Prepare Datasets ([Jupyter notebook](notebooks/prepare_datasets.ipynb))

We develop our method on these datasets:
- [KITTI object detection 3D dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
- [nuScenes dataset v1.0](https://www.nuscenes.org/nuscenes)
- [Waymo dataset v1.0](https://waymo.com/open/data/)

1. Configure `dataset_path` in [config_path.py](config_path.py).

    Raw datasets will be organized as the following structure:
    
    <pre>
    dataset_path/
        | kitti/               # KITTI object detection 3D dataset
            | training/
            | testing/
        | nusc/                # nuScenes dataset v1.0
            | maps/
            | samples/
            | sweeps/
            | v1.0-trainval/
        | waymo/               # Waymo dataset v1.0
            | training/
            | validation/
    </pre>

2. Download all datasets.
For performance and bias reasons, we subsample the NuScenes and Waymo Open Dataset to the approximate number of frames of the KITTI dataset. 
We choose a random selection of both datasets, so that we receive 7600 test-frames, 3750 training-frames and 3800 validation-frames

For Waymo:
  Download all files
    Treat validation as sample pool for test dataset
      7600 test / 200 = ca 38 files

    ```bash
    rm -f $(find . -type f | shuf -n 152)
    ```

    Treat Training as Sample pool for Train + Validation
    (3750 train + 3800 val) / 200 = 38 files
    
    ```bash
    rm -f $(find . -type f | shuf -n 748)
    ```

For NuScenes:
  Treat Training as Sample pool for Test Dataset
    7600 test / 40 = ca 190 scenes
    download 4 files 

  Treat Training as Sample pool for Train + Validation
    (3750 train + 3800 val) / 40 = ca 190 scenes  round up to 340
    download 4 files

3. Convert all datasets to `KITTI format`.

    ```bash
    cd scripts/
    python -m pip install -r convert_requirements.txt
    python convert.py [--datasets argo+nusc+lyft+waymo]
    ```

4. Split validation set
    Generate Waymo Split:
    ```bash
    #    choose 7600 test-frames, 3750 training-frames and 3800
    #random list of generated frames:
    l = list(range(0,9929))
    l_train = random.sample(l, 3750)
    l_2 = [x for x in l if x not in l_train]
    l_val = random.sample(l_2, 3800)

    #fixed width for document:
    l_train = ['{0:06}'.format(x) for x in l_train]
    l_val = ['{0:06}'.format(x) for x in l_val]

    #write train- and validaton-file:
    with open('/scratch/ws/1/s1510289-KITTI/train.txt', mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(l_train))

    with open('/scratch/ws/1/s1510289-KITTI/val.txt', mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(l_val))

    #write test-file:
    l_3 = list(range(0,9923))
    l_test = random.sample(l_3, 7600)
    l_test = ['{0:06}'.format(x) for x in l_test]

    with open('/scratch/ws/1/s1510289-KITTI/test.txt', mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(l_test))
    ```
    
    Generate nuScenes Split:
    ```bash    
    #choose 7600 test-frames, 3750 training-frames and 3800:
    #training: 8573 only even number 0, 2, 4 … , 17144
    #testing: 8572 only odd numbers 1 , 3, … , 17143

    #random list of generated frames:
    l_all = list(range(0,17145))
    l = [x for x in l_all if x % 2 == 0]
    l_train = random.sample(l, 3750)
    l_2 = [x for x in l if x not in l_train]
    l_val = random.sample(l_2, 3800)

    #fixed width for document:
    l_train = ['{0:06}'.format(x) for x in l_train]
    l_val = ['{0:06}'.format(x) for x in l_val]

    #write train- and validation-file:
    with open('/scratch/ws/1/s1510289-KITTI/train.txt', mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(l_train))

    with open('/scratch/ws/1/s1510289-KITTI/val.txt', mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(l_val))

    #write test-file:
    l_all = list(range(0,17145))
    l_3 = [x for x in l_all if x % 2 == 1]
    l_test = random.sample(l_3, 7600)
    l_test = ['{0:06}'.format(x) for x in l_test]

    with open('/scratch/ws/1/s1510289-KITTI/test.txt', mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(l_test))

    We provide the `train`/`val` split used in our experiments under [split](split/) folder.
    
    ```bash
    cd split/
    python replace_split.py
    ```

   
4. Generate `car` subset

    We filter scenes and only keep those with cars.
    
    ```bash
    cd scripts/
    python gen_car_split.py
    ```

### Statistical Normalization

1. Compute car size statistics of each dataset. 
The computed statistics are stored as `label_stats_{train/val/test}.json` under KITTI format dataset root.

    ```bash
    cd stat_norm/
    python stat.py
    ```

2. Generate rescaled datasets according to car size statistics. 
The rescaled datasets are stored under `$dataset_path/rescaled_datasets` by default.

    ```bash
    cd stat_norm/
    python norm.py [--path $PATH]
    ```
   
### Training/ Fine-Tune

We use [PointRCNN](https://arxiv.org/abs/1812.04244) to validate our method. 

1. Setup PointRCNN

    ```bash
    cd pointrcnn/
    ./build_and_install.sh
    ```

2. Build datasets in PointRCNN format.

    ```bash
    cd pointrcnn/tools/
    python generate_multi_data.py
    python generate_gt_database.py --root ...
    ```
   The `NuScence` dataset has much less points in each bounding box, so we have to turn of the `GT_AUG_HARD_RATIO` augmentation.

3. Download the models pretrained on source domains of [Wang et al. (2020)](https://arxiv.org/abs/2005.08139) from [google drive](https://drive.google.com/drive/folders/14MXjNImFoS2P7YprLNpSmFBsvxf5J2Kw?usp=sharing) using [gdrive](https://github.com/gdrive-org/gdrive/releases/download/2.1.0/gdrive-linux-x64).

    ```bash
    cd pointrcnn/tools/
    gdrive download -r 14MXjNImFoS2P7YprLNpSmFBsvxf5J2Kw
    ```
    
4. Adapt to a new domain by re-training with rescaled data.

    ```bash
    cd pointrcnn/tools/
    
    python train_rcnn.py --cfg_file ...
    srun python train_rcnn.py --train_mode rcnn --batch_size $BATCH_SIZE --epochs $EPOCHS --root /pointrcnn/multi_data/$DATASET --cfg_file cfgs/CFG_FILE_YAML --gt_database gt_database/$GT_DATABASE_PKL --output_dir /pointrcnn/output/training/$OUTPUT_DIR --ckpt /pointrcnn/pretrained_ckpt/CKPT_PTH [--subsample $NUMBER_OF_SUBSAMPLE --shuffle_subsample [True/False]]
    ```

### Self-Training
1. Inference of source model on target data
2. Sampling of pseudo-labels
```bash
cd pipeline_scripts/self_training
python inference_sample.py $INFERENCE_DIR_PATH $PSEUDO_LABEL_OUTPUT_FILE
```

3. Moving pseudo-labels to multidata
```bash
cd pipeline_scripts/self_training
./mv_inf_sample_to_multidata.sh $INFERNCE_SAMPLE_TEXTFILE $SOURCE_MODEL $TARGET_DATASET
```

4. Generate new ground-truth
```bash
cd pointrcnn/tools/
srun python generate_gt_database.py --root ./pointrcnn/multi_data/$DATASET --save_dir ./gt_database/$SAVE_DIR --self_training $GT_DATABASE_FILENAME --self_training_textfile SELFTRAINING_SAMPLE_TEXTFILE
```

5. Training

```bash
cd pointrcnn/tools
srun python train_rcnn.py --train_mode rcnn --batch_size $BATCH_SIZE --epochs $EPOCHS --root /pointrcnn/multi_data/$DATASET --cfg_file cfgs/CFG_FILE_YAML --gt_database gt_database/$GT_DATABASE_FILENAME --output_dir /pointrcnn/output/training/$OUTPUT_DIR --ckpt /pointrcnn/pretrained_ckpt/SELFTRAINING_CKPT_PTH [--subsample $NUMBER_OF_SUBSAMPLE --shuffle_subsample [True/False]] --self_training $PSEUDO_LABEL_DIR --self_training_textfile $INFERNCE_SAMPLE_TEXTFILE
```

6. Inference
7. Evaluation


### Inference
```bash
cd pointrcnn/tools/
python eval_rcnn.py --ckpt /path/to/checkpoint.pth --dataset $dataset --output_dir $output_dir 
```

### Evaluation

We provide [evaluation code](evaluate/evaluate.py#L279) with
- old (based on bbox height) and new (based on distance) difficulty metrics
- <em>output transformation</em> functions to locate domain gap

```bash
cd evaluate/
python evaluate.py --result_path $predictions --dataset_path $dataset_root --metric [old/new]
```

Read all evaluation results and write them into .csv-file:
```bash
cd /pipeline_scripts/evaluation
python output_read.py $dir_name $output_path §filename
```
