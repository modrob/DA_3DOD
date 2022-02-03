import os

dataset_path = os.path.expanduser("/scratch/ws/1/s1510289-KITTI/3D_adapt_auto_driving/scratch/driving_datasets")

print(f"Datasets will be stored in {dataset_path}")
assert dataset_path, "Please set dataset path in config_path.py"

os.makedirs(dataset_path, exist_ok=True)

# Dictionary of raw dataset paths
raw_path_dic = {
    "kitti": os.path.join(dataset_path, "kitti"),
    "argo": os.path.join(dataset_path, "argo"),
    "nusc": os.path.join(dataset_path, "nusc"),
    "lyft": os.path.join(dataset_path, "lyft"),
    "waymo": os.path.join(dataset_path, "waymo"),
    "argo-in-kitti-format": os.path.join(dataset_path, "argo-in-kitti-format"),
    "nusc-in-kitti-format": os.path.join(dataset_path, "nusc-in-kitti-format"),
    "lyft-in-kitti-format": os.path.join(dataset_path, "lyft-in-kitti-format"),
    "waymo-in-kitti-format": os.path.join(dataset_path, "waymo-in-kitti-format"),
}

# KITTI format dataset paths used in experiments
dataset_paths = {
    #"kitti": os.path.join(dataset_path, "kitti"),
    #"waymo": os.path.join(dataset_path, "waymo-in-kitti-format"), 
    #"nusc": os.path.join(dataset_path, "nusc-in-kitti-format"),
    "kitti_to_nusc": os.path.join(dataset_path, "rescaled_datasets/kitti_scaledto_nusc"),
    "kitti_to_waymo": os.path.join(dataset_path, "rescaled_datasets/kitti_scaledto_waymo"),
    "nusc_to_kitti": os.path.join(dataset_path, "rescaled_datasets/nusc_scaledto_kitti"),
    "nusc_to_waymo": os.path.join(dataset_path, "rescaled_datasets/nusc_scaledto_waymo"),
    "waymo_to_kitti": os.path.join(dataset_path, "rescaled_datasets/waymo_scaledto_kitti"),
    "waymo_to_nusc": os.path.join(dataset_path, "rescaled_datasets/waymo_scaledto_nusc"),
}

dataset_full_name = {
    "kitti": "KITTI",
    "argo": "Argoverse",
    "nusc": "nuScenes",
    "lyft": "Lyft",
    "waymo": "Waymo",
}

datasets = list(dataset_paths.keys())
