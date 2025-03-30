#!/usr/bin/env python
"""
Setup script for OpenPCDet with CIE dataset.
This script performs the following steps:
1. Clone OpenPCDet repository if it doesn't exist
2. Install OpenPCDet dependencies
3. Set up CIE dataset for OpenPCDet
4. Copy configuration files to OpenPCDet directory
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path
import time
import numpy as np
import pickle


def run_command(command, cwd=None, shell=False):
    """Run a command and print its output in real-time."""
    print(f"Running: {command}")

    python_path = sys.executable
    if isinstance(command, str) and not shell:
        command = command.split()

    # Ifthe command starts with python, replace it with the virtual env Python
    if isinstance(command, list) and len(command) > 0 and command[0] == "python":
        command = [python_path, "-m", "pip"] + command[1:]
    elif isinstance(command, str) and command.startsswith("python "):
        command = f"{python_path} -m pip {command[4:]}"

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        cwd=cwd,
        shell=shell,
    )

    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
        return False

    return True


def clone_openpcdet(target_dir):
    """Clone OpenPCDet repository."""
    if os.path.exists(os.path.join(target_dir, "OpenPCDet")):
        print(f"OpenPCDet already exists in {target_dir}")
        return True

    print(f"Cloning OpenPCDet to {target_dir}...")
    return run_command(
        f"git clone https://github.com/open-mmlab/OpenPCDet.git {os.path.join(target_dir, 'OpenPCDet')}"
    )


def install_dependencies(openpcdet_dir):
    """Install OpenPCDet dependencies."""
    print("Installing OpenPCDet dependencies...")

    # Install basic requirements
    requirements_file = os.path.join(openpcdet_dir, "requirements.txt")
    if os.path.exists(requirements_file):
        if not run_command(["pip", "install", "-r", requirements_file]):
            return False
    else:
        print(f"Requirements file not found: {requirements_file}")
        return False

    # Check for setup.py in tools directory
    tools_setup_py = os.path.join(openpcdet_dir, "tools", "setup.py")
    root_setup_py = os.path.join(openpcdet_dir, "setup.py")

    if os.path.exists(tools_setup_py) and not os.path.exists(root_setup_py):
        print("Found setup.py in tools directory, creating symbolic link...")
        os.symlink(tools_setup_py, root_setup_py)

    # Install OpenPCDet
    if os.path.exists(root_setup_py):
        if not run_command("{sys.executable} - m pip install -e .", cwd=openpcdet_dir):
            return False
    else:
        print("setup.py not found, installation may not be complete")
        print("Continuing with setup anyway...")

    return True


def setup_cie_dataset(openpcdet_dir, cie_data_dir, output_dir):
    """Setup CIE dataset for OpenPCDet."""
    print("Setting up CIE dataset...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Write the CIE dataset class
    cie_dataset_file = os.path.join(output_dir, "cie_dataset.py")

    with open(cie_dataset_file, "w") as f:
        f.write(
            """
import os
import numpy as np
import torch
import pickle
from pathlib import Path
from ..dataset import DatasetTemplate

class CIEDataset(DatasetTemplate):
    \"\"\"Custom dataset class for CIE point cloud data.\"\"\"
    
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger
        )
        
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path
        
        split_dir = os.path.join(self.root_path, 'ImageSets', (self.split + '.txt'))
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if os.path.exists(split_dir) else [0]
        
        self.infos = []
        self.include_data(self.mode)
        
    def include_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading CIE dataset')
        
        cie_infos = []
        
        info_path = self.dataset_cfg.INFO_PATH[mode]
        info_path = [os.path.join(self.root_path, path) for path in info_path]
        
        for path in info_path:
            if not os.path.exists(path):
                self.logger.info(f'Info file {path} not found, creating empty info')
                # Create an empty info file if it doesn't exist
                bin_path = os.path.join(self.root_path, 'cie.bin')
                if not os.path.exists(bin_path):
                    # If no data exists yet, create an empty one
                    pcd_path = os.path.join(self.root_path, 'CIE.pcd')
                    if os.path.exists(pcd_path):
                        import open3d as o3d
                        pcd = o3d.io.read_point_cloud(pcd_path)
                        points = np.asarray(pcd.points)
                        if pcd.has_colors():
                            colors = np.asarray(pcd.colors)
                            intensity = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]
                        else:
                            intensity = np.ones(len(points))
                        
                        points_with_intensity = np.column_stack((points, intensity))
                        points_with_intensity.astype(np.float32).tofile(bin_path)
                
                empty_info = {
                    'point_cloud': {
                        'lidar_idx': 'cie',
                        'num_features': 4,
                        'num_points': 0
                    },
                    'annos': {
                        'name': [],
                        'boxes_lidar': np.zeros((0, 7)),
                        'difficulty': [],
                        'score': [],
                        'num_points_in_gt': []
                    },
                    'frame_id': 0,
                    'metadata': {
                        'image_path': None,
                        'lidar_path': bin_path if os.path.exists(bin_path) else None
                    }
                }
                
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'wb') as f:
                    pickle.dump([empty_info], f)
            
            with open(path, 'rb') as f:
                infos = pickle.load(f)
                cie_infos.extend(infos)
        
        self.infos.extend(cie_infos)
        
        if self.logger is not None:
            self.logger.info(f'Total samples for CIE dataset {mode}: {len(cie_infos)}')
    
    def get_lidar(self, idx):
        lidar_file = self.infos[idx]['metadata']['lidar_path'] if 'metadata' in self.infos[idx] else None
        
        if lidar_file and os.path.exists(lidar_file):
            return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        else:
            # If no specific lidar file, use the default one
            bin_path = os.path.join(self.root_path, 'cie.bin')
            if os.path.exists(bin_path):
                return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
            else:
                # If no bin file, try to read PCD
                pcd_path = os.path.join(self.root_path, 'CIE.pcd')
                if os.path.exists(pcd_path):
                    import open3d as o3d
                    pcd = o3d.io.read_point_cloud(pcd_path)
                    points = np.asarray(pcd.points)
                    
                    # Add intensity
                    if pcd.has_colors():
                        colors = np.asarray(pcd.colors)
                        intensity = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]
                    else:
                        intensity = np.ones(len(points))
                    
                    return np.column_stack((points, intensity)).astype(np.float32)
                else:
                    raise FileNotFoundError(f"No point cloud data found for sample {idx}")
    
    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        import concurrent.futures as futures
        
        def process_single_scene(sample_idx):
            print(f'Processing sample index: {sample_idx}')
            
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': 'cie'}
            info['point_cloud'] = pc_info
            
            # For CIE dataset, we'll set static metadata paths
            info['metadata'] = {
                'image_path': None,
                'lidar_path': os.path.join(self.root_path, 'cie.bin')
            }
            
            # Empty annotations for now (to be filled when annotations are available)
            if has_label:
                annotations = {
                    'name': [],
                    'boxes_lidar': np.zeros((0, 7)),
                    'difficulty': [],
                    'score': [],
                    'num_points_in_gt': []
                }
                info['annos'] = annotations
            
            info['frame_id'] = sample_idx
            return info
        
        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        
        # Process in parallel
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = list(executor.map(process_single_scene, sample_id_list))
        
        return infos
    
    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch
        
        if info_path is None:
            info_path = os.path.join(self.root_path, f'cie_infos_{split}.pkl')
        
        if not os.path.exists(info_path):
            print(f'No info file found at {info_path}')
            return
        
        database_save_path = os.path.join(self.root_path, f'gt_database_{split}')
        db_info_save_path = os.path.join(self.root_path, f'cie_dbinfos_{split}.pkl')
        
        os.makedirs(database_save_path, exist_ok=True)
        
        print(f'Creating groundtruth database for {split} set')
        
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        
        # For CIE dataset, we don't have ground truth annotations yet
        # This function will need to be updated when annotations are available
        all_db_infos = {}
        
        # Just create an empty DB info file for now
        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)
        
        print(f'Database creation finished (empty for now)')
    
    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        \"\"\"
        Generate prediction results in KITTI format.
        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from model
            class_names: list of class names
            output_path: if provided, save the results to this path
        Returns:
            pred_dicts: list of predictions for each sample
        \"\"\"
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7]),
                'score': np.zeros(num_samples),
                'pred_labels': np.zeros(num_samples)
            }
            return ret_dict
        
        def generate_single_sample_dict(batch_index, box_dict):
            pred_dict = get_template_prediction(len(box_dict['pred_boxes']))
            if len(box_dict['pred_boxes']) == 0:
                return pred_dict
            
            pred_dict['name'] = np.array(class_names)[box_dict['pred_labels'].cpu() - 1]
            pred_dict['boxes_lidar'] = box_dict['pred_boxes'].cpu().numpy()
            pred_dict['score'] = box_dict['pred_scores'].cpu().numpy()
            pred_dict['pred_labels'] = box_dict['pred_labels'].cpu().numpy()
            
            return pred_dict
        
        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]
            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)
            
            if output_path is not None:
                os.makedirs(output_path, exist_ok=True)
                cur_det_file = os.path.join(output_path, f'{frame_id}.txt')
                with open(cur_det_file, 'w') as f:
                    for idx in range(len(single_pred_dict['name'])):
                        box = single_pred_dict['boxes_lidar'][idx]
                        score = single_pred_dict['score'][idx]
                        name = single_pred_dict['name'][idx]
                        box_str = ' '.join([str(item) for item in box])
                        line = f'{name} {score:.4f} {box_str}\\n'
                        f.write(line)
        
        return annos
    
    def evaluation(self, det_annos, class_names, **kwargs):
        \"\"\"
        CIE dataset evaluation (to be implemented when we have ground truth).
        For now, we'll just return empty metrics.
        \"\"\"
        print('Evaluating CIE dataset (placeholder for now)')
        
        # Placeholder evaluation metrics
        ap_result_str = ''
        ap_dict = {}
        
        for class_idx, class_name in enumerate(class_names):
            ap_dict[class_name] = {
                'precision': np.array([0, 0, 0]),
                'recall': np.array([0, 0, 0]),
                'aos': np.array([0, 0, 0]),
                'gt_num': 0
            }
            ap_result_str += f'\\n{class_name} AP: {0:.4f} {0:.4f} {0:.4f}\\n'
        
        print(ap_result_str)
        
        return {
            'recall': 0,
            'precision': 0,
            'mAP': 0
        }
        
    def __len__(self):
        return len(self.infos) if self.infos else 1
    
    def __getitem__(self, index):
        if self.infos:
            info = self.infos[index]
        else:
            # If no infos, create a dummy one
            info = {
                'point_cloud': {'num_features': 4, 'lidar_idx': 'cie'},
                'metadata': {'lidar_path': os.path.join(self.root_path, 'cie.bin')},
                'frame_id': 0
            }
        
        sample_idx = info['frame_id']
        points = self.get_lidar(index)
        
        input_dict = {
            'points': points,
            'frame_id': sample_idx
        }
        
        # Add annotations if available
        if 'annos' in info:
            annos = info['annos']
            if annos and len(annos['name']) > 0:
                gt_names = annos['name']
                gt_boxes = annos['boxes_lidar']
                input_dict.update({
                    'gt_names': gt_names,
                    'gt_boxes': gt_boxes
                })
        
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
"""
        )

    # Copy the dataset file to OpenPCDet
    pcdet_dataset_dir = os.path.join(openpcdet_dir, "pcdet", "datasets", "cie")
    os.makedirs(pcdet_dataset_dir, exist_ok=True)
    shutil.copy(cie_dataset_file, os.path.join(pcdet_dataset_dir, "cie_dataset.py"))

    # Create __init__.py file
    with open(os.path.join(pcdet_dataset_dir, "__init__.py"), "w") as f:
        f.write(
            "from .cie_dataset import CIEDataset\n\n__all__ = {\n    'CIEDataset': CIEDataset\n}\n"
        )

    # Update the dataset init file to include CIE dataset
    dataset_init_file = os.path.join(openpcdet_dir, "pcdet", "datasets", "__init__.py")
    with open(dataset_init_file, "r") as f:
        content = f.read()

    if "CIEDataset" not in content:
        with open(dataset_init_file, "w") as f:
            content = content.replace(
                "from pcdet.datasets.kitti.kitti_dataset import KittiDataset",
                "from pcdet.datasets.kitti.kitti_dataset import KittiDataset\nfrom pcdet.datasets.cie.cie_dataset import CIEDataset",
            )
            content = content.replace(
                "'KittiDataset': KittiDataset,",
                "'KittiDataset': KittiDataset,\n    'CIEDataset': CIEDataset,",
            )
            f.write(content)

    # Create directory structure for CIE dataset
    cie_data_root = os.path.join(cie_data_dir)
    os.makedirs(os.path.join(cie_data_root, "ImageSets"), exist_ok=True)

    # Create train/test split files
    with open(os.path.join(cie_data_root, "ImageSets", "train.txt"), "w") as f:
        f.write("0\n")  # Single sample for now

    with open(os.path.join(cie_data_root, "ImageSets", "test.txt"), "w") as f:
        f.write("0\n")  # Single sample for now

    # Copy config file
    config_dir = os.path.join(openpcdet_dir, "tools", "cfgs", "cie_models")
    os.makedirs(config_dir, exist_ok=True)

    config_file = os.path.join(output_dir, "cie_pointpillar.yaml")
    with open(config_file, "w") as f:
        f.write(
            """CLASS_NAMES: ['Chair', 'Table', 'Monitor', 'Projection_Screen', 
              'Projector', 'TV', 'Computer', 'Misc']

DATA_CONFIG:
    _BASE_CONFIG_: cfgs/dataset_configs/kitti_dataset.yaml
    DATASET: 'CIEDataset'
    DATA_PATH: '../data/cie'
    
    POINT_CLOUD_RANGE: [-50, -50, -5, 50, 50, 5]
    
    DATA_SPLIT: {
        'train': train,
        'test': test
    }
    
    INFO_PATH: {
        'train': [cie_infos_train.pkl],
        'test': [cie_infos_test.pkl],
    }
    
    FOV_POINTS_ONLY: False
    
    DATA_AUGMENTOR:
        DISABLE_AUG_LIST: ['gt_sampling']
        AUG_CONFIG_LIST:
            - NAME: random_world_flip
              ALONG_AXIS_LIST: ['x', 'y']

            - NAME: random_world_rotation
              WORLD_ROT_ANGLE: [-0.78539816, 0.78539816]

            - NAME: random_world_scaling
              WORLD_SCALE_RANGE: [0.95, 1.05]
    
    POINT_FEATURE_ENCODING: {
        encoding_type: absolute_coordinates_encoding,
        used_feature_list: ['x', 'y', 'z', 'intensity'],
        src_feature_list: ['x', 'y', 'z', 'intensity'],
    }
    
    DATA_PROCESSOR:
        - NAME: mask_points_and_boxes_outside_range
          REMOVE_OUTSIDE_BOXES: True

        - NAME: shuffle_points
          SHUFFLE_ENABLED: {
            'train': True,
            'test': False
          }

        - NAME: transform_points_to_voxels
          VOXEL_SIZE: [0.05, 0.05, 0.1]
          MAX_POINTS_PER_VOXEL: 10
          MAX_NUMBER_OF_VOXELS: {
            'train': 80000,
            'test': 90000
          }

MODEL:
    NAME: PointPillar

    VFE:
        NAME: PillarVFE
        WITH_DISTANCE: False
        USE_ABSLOTE_XYZ: True
        USE_NORM: True
        NUM_FILTERS: [64, 64]

    MAP_TO_BEV:
        NAME: PointPillarScatter
        NUM_BEV_FEATURES: 64

    BACKBONE_2D:
        NAME: BaseBEVBackbone
        LAYER_NUMS: [3, 5, 5]
        LAYER_STRIDES: [2, 2, 2]
        NUM_FILTERS: [64, 128, 256]
        UPSAMPLE_STRIDES: [1, 2, 4]
        NUM_UPSAMPLE_FILTERS: [128, 128, 128]

    DENSE_HEAD:
        NAME: AnchorHeadSingle
        CLASS_AGNOSTIC: False

        USE_DIRECTION_CLASSIFIER: True
        DIR_OFFSET: 0.78539
        DIR_LIMIT_OFFSET: 0.0
        NUM_DIR_BINS: 2

        ANCHOR_GENERATOR_CONFIG: [
            {
                'class_name': 'Chair',
                'anchor_sizes': [[0.6, 0.6, 1.0]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-0.6],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.6,
                'unmatched_threshold': 0.45
            },
            {
                'class_name': 'Table',
                'anchor_sizes': [[1.0, 1.0, 0.8]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-0.6],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.55,
                'unmatched_threshold': 0.4
            },
            {
                'class_name': 'Monitor',
                'anchor_sizes': [[0.5, 0.5, 0.5]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [0.0],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.55,
                'unmatched_threshold': 0.4
            },
            {
                'class_name': 'Projection_Screen',
                'anchor_sizes': [[2.0, 0.2, 1.5]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [0.0],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.55,
                'unmatched_threshold': 0.4
            },
            {
                'class_name': 'Projector',
                'anchor_sizes': [[0.4, 0.4, 0.3]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [1.5],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.5,
                'unmatched_threshold': 0.35
            },
            {
                'class_name': 'TV',
                'anchor_sizes': [[1.0, 0.2, 0.6]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [0.5],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.5,
                'unmatched_threshold': 0.35
            },
            {
                'class_name': 'Computer',
                'anchor_sizes': [[0.6, 0.6, 0.6]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [0.0],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.5,
                'unmatched_threshold': 0.35
            },
            {
                'class_name': 'Misc',
                'anchor_sizes': [[0.5, 0.5, 0.5]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [0.0],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.5,
                'unmatched_threshold': 0.35
            }
        ]

        TARGET_ASSIGNER_CONFIG:
            NAME: AxisAlignedTargetAssigner
            POS_FRACTION: -1.0
            SAMPLE_SIZE: 512
            NORM_BY_NUM_EXAMPLES: False
            MATCH_HEIGHT: False
            BOX_CODER: ResidualCoder

        LOSS_CONFIG:
            LOSS_WEIGHTS: {
                'cls_weight': 1.0,
                'loc_weight': 2.0,
                'dir_weight': 0.2,
                'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }

    POST_PROCESSING:
        RECALL_THRESH_LIST: [0.3, 0.5, 0.7]
        SCORE_THRESH: 0.1
        OUTPUT_RAW_SCORE: False

        EVAL_METRIC: kitti

        NMS_CONFIG:
            MULTI_CLASSES_NMS: False
            NMS_TYPE: nms_gpu
            NMS_THRESH: 0.01
            NMS_PRE_MAXSIZE: 4096
            NMS_POST_MAXSIZE: 500


OPTIMIZATION:
    BATCH_SIZE_PER_GPU: 4
    NUM_EPOCHS: 80

    OPTIMIZER: adam_onecycle
    LR: 0.003
    WEIGHT_DECAY: 0.01
    MOMENTUM: 0.9

    MOMS: [0.95, 0.85]
    PCT_START: 0.4
    DIV_FACTOR: 10
    DECAY_STEP_LIST: [35, 45]
    LR_DECAY: 0.1
    LR_CLIP: 0.0000001

    LR_WARMUP: False
    WARMUP_EPOCH: 1

    GRAD_NORM_CLIP: 10
"""
        )

    shutil.copy(config_file, os.path.join(config_dir, "cie_pointpillar.yaml"))

    print(f"CIE dataset setup completed: {cie_data_root}")
    return True


def download_pretrained_model(openpcdet_dir, output_dir):
    """Download a pretrained model from OpenPCDet."""
    print("Downloading a pretrained model...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # We'll use the KITTI PointPillar model as a starting point
    model_url = "https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view"

    print(f"Please download the PointPillar model manually from {model_url}")
    print(f"Save it to {output_dir}/pointpillar_7728.pth")

    # Since Google Drive direct download is complex, we'll prompt the user to download manually
    response = input("Have you downloaded the model? (y/n): ")

    if response.lower() == "y":
        if os.path.exists(os.path.join(output_dir, "pointpillar_7728.pth")):
            print("Model file found!")
            return True
        else:
            print(f"Model file not found in {output_dir}")
            return False
    else:
        print("Please download the model to continue.")
        return False


def process_cie_data(input_dir, output_dir):
    """Process CIE data for OpenPCDet."""
    print("Processing CIE data...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Merge PCD files if needed
    merged_pcd_path = os.path.join(output_dir, "CIE.pcd")
    if not os.path.exists(merged_pcd_path):
        print("Merging PCD files...")
        # Import our merging script
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from merge_colab_script import merge_pcd_files

        merge_pcd_files(input_dir, merged_pcd_path)

    # Create empty files needed by OpenPCDet
    for split in ["train", "test"]:
        info_path = os.path.join(output_dir, f"cie_infos_{split}.pkl")
        if not os.path.exists(info_path):
            print(f"Creating empty info file for {split} split...")
            info = {
                "point_cloud": {"lidar_idx": "cie", "num_features": 4, "num_points": 0},
                "annos": {
                    "name": [],
                    "boxes_lidar": np.zeros((0, 7)),
                    "difficulty": [],
                    "score": [],
                    "num_points_in_gt": [],
                },
                "frame_id": 0,
                "metadata": {
                    "image_path": None,
                    "lidar_path": os.path.join(output_dir, "cie.bin"),
                },
            }

            with open(info_path, "wb") as f:
                pickle.dump([info], f)

    # For OpenPCDet compatibility, also save as bin file
    bin_path = os.path.join(output_dir, "cie.bin")
    if not os.path.exists(bin_path):
        print("Converting PCD to binary format...")
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(merged_pcd_path)
        points = np.asarray(pcd.points)

        # Extract colors as intensity if available
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            intensity = (
                0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]
            )
        else:
            intensity = np.ones(len(points))

        # Create point cloud data with intensity
        points_with_intensity = np.column_stack((points, intensity))
        points_with_intensity.astype(np.float32).tofile(bin_path)

    print(f"CIE data processed and saved to {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Setup OpenPCDet for CIE Dataset")
    parser.add_argument(
        "--openpcdet_dir",
        type=str,
        default="./OpenPCDet",
        help="Directory for OpenPCDet",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory with CIE PCD files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/cie",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory for model checkpoints",
    )
    parser.add_argument(
        "--skip_clone", action="store_true", help="Skip cloning OpenPCDet"
    )
    parser.add_argument(
        "--skip_install", action="store_true", help="Skip installing dependencies"
    )
    parser.add_argument(
        "--skip_process", action="store_true", help="Skip processing CIE data"
    )
    parser.add_argument(
        "--skip_model", action="store_true", help="Skip downloading model"
    )
    args = parser.parse_args()

    # Create directories
    os.makedirs(args.openpcdet_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Step 1: Clone OpenPCDet
    if not args.skip_clone:
        if not clone_openpcdet(os.path.dirname(args.openpcdet_dir)):
            print("Failed to clone OpenPCDet")
            return False

    # Step 2: Install dependencies
    if not args.skip_install:
        if not install_dependencies(args.openpcdet_dir):
            print("Failed to install dependencies")
            return False

    # Step 3: Process CIE data
    if not args.skip_process:
        if not process_cie_data(args.input_dir, args.output_dir):
            print("Failed to process CIE data")
            return False

    # Step 4: Setup CIE dataset for OpenPCDet
    if not setup_cie_dataset(args.openpcdet_dir, args.output_dir, args.output_dir):
        print("Failed to setup CIE dataset")
        return False

    # Step 5: Download pretrained model
    if not args.skip_model:
        if not download_pretrained_model(args.openpcdet_dir, args.checkpoint_dir):
            print("Failed to download pretrained model")
            return False

    print("\n=== OpenPCDet Setup Complete ===")
    print(f"OpenPCDet directory: {args.openpcdet_dir}")
    print(f"CIE data directory: {args.output_dir}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")

    print("\nTo run detection with OpenPCDet, use:")
    print(f"cd {args.openpcdet_dir}")
    print(
        f"python tools/demo.py --cfg_file tools/cfgs/cie_models/cie_pointpillar.yaml --ckpt {args.checkpoint_dir}/pointpillar_7728.pth --data_path {args.output_dir}/CIE.pcd"
    )

    print("\nTo train a model for CIE dataset, first create annotations and then run:")
    print(
        f"python tools/train.py --cfg_file tools/cfgs/cie_models/cie_pointpillar.yaml"
    )

    return True


if __name__ == "__main__":
    start_time = time.time()
    success = main()
    end_time = time.time()

    print(
        f"\nSetup {'completed' if success else 'failed'} in {end_time - start_time:.2f} seconds"
    )
    sys.exit(0 if success else 1)
