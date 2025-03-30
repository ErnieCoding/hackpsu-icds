#!/usr/bin/env python
"""
Run Object Detection on CIE Point Cloud using OpenPCDet.
This script:
1. Loads a merged PCD file
2. Runs detection using OpenPCDet model
3. Visualizes the results
"""

import os
import sys
import numpy as np
import pickle
import argparse
import open3d as o3d
import torch
import tempfile
import subprocess
from pathlib import Path

def ensure_openpcdet_available():
    """Check if OpenPCDet is available and accessible."""
    try:
        # Try to import OpenPCDet
        sys.path.insert(0, './OpenPCDet')
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.utils import common_utils
        return True
    except ImportError:
        print("OpenPCDet not found or not properly installed.")
        print("Please run setup_openpcdet.py first.")
        return False

def run_demo_script(openpcdet_dir, cfg_file, ckpt, data_path, output_dir):
    """Run OpenPCDet's demo script directly."""
    print(f"Running OpenPCDet demo script...")
    
    # Make paths absolute
    cfg_file = os.path.abspath(cfg_file)
    ckpt = os.path.abspath(ckpt)
    data_path = os.path.abspath(data_path)
    output_dir = os.path.abspath(output_dir)
    
    # Extract model name from cfg_file for folder organization
    cfg_basename = os.path.basename(cfg_file)
    model_name = os.path.splitext(cfg_basename)[0]
    
    # Create model-specific output directory
    model_output_dir = os.path.join(output_dir, f"openpcdet_{model_name}")
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Change to OpenPCDet directory
    original_dir = os.getcwd()
    os.chdir(openpcdet_dir)
    os.chdir(openpcdet_dir)
    
    # Run the demo script
    cmd = [
        sys.executable, 'tools/demo.py',
        '--cfg_file', cfg_file,
        '--ckpt', ckpt,
        '--data_path', data_path,
        '--ext', '.pcd',
        '--save_point_cloud',
        '--output_dir', model_output_dir
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        success = True
    except subprocess.CalledProcessError as e:
        print(f"Error running demo script: {e}")
        success = False
    
    # Return to original directory
    os.chdir(original_dir)
    
    return success, model_output_dir

def visualize_results(pcd_path, result_path, output_image=None):
    """
    Visualize detection results.
    
    Args:
        pcd_path: Path to point cloud file
        result_path: Path to detection results file
        output_image: Path to save visualization image (optional)
    """
    print(f"Visualizing detection results...")
    
    # Load point cloud
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    # Load detection results
    with open(result_path, 'rb') as f:
        results = pickle.load(f)
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="OpenPCDet Detection Results", width=1280, height=720)
    
    # Add point cloud
    vis.add_geometry(pcd)
    
    # Color map for different classes
    colors = {
        'Chair': [1, 0, 0],         # Red
        'Table': [0, 1, 0],         # Green
        'Monitor': [0, 0, 1],       # Blue
        'Projection_Screen': [1, 1, 0],  # Yellow
        'Projector': [1, 0, 1],     # Magenta
        'TV': [0, 1, 1],            # Cyan
        'Computer': [0.5, 0.5, 0],  # Olive
        'Misc': [0.5, 0.5, 0.5]     # Gray
    }
    
    # Extract information
    boxes = results.get('boxes_lidar', [])
    labels = results.get('labels', [])
    scores = results.get('scores', [])
    class_names = results.get('class_names', [])
    
    # Add bounding boxes
    for i, box in enumerate(boxes):
        x, y, z, dx, dy, dz, heading = box
        
        # Create oriented bounding box
        obb = o3d.geometry.OrientedBoundingBox(
            center=[x, y, z],
            R=o3d.geometry.get_rotation_matrix_from_xyz([0, 0, heading]),
            extent=[dx, dy, dz]
        )
        
        # Get class name
        class_name = class_names[i] if i < len(class_names) else 'Unknown'
        
        # Set color
        color = colors.get(class_name, [0.5, 0.5, 0.5])
        obb.color = color
        
        # Add to visualizer
        vis.add_geometry(obb)
    
    # Set rendering options
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.point_size = 2.0
    
    # Show visualization
    vis.run()
    
    # Save screenshot if requested
    if output_image:
        vis.capture_screen_image(output_image)
        print(f"Visualization saved to {output_image}")
    
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description="Run Object Detection on CIE Point Cloud using OpenPCDet")
    parser.add_argument('--openpcdet_dir', type=str, default='./OpenPCDet', help='OpenPCDet directory')
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/cie_models/cie_pointpillar.yaml', 
                        help='Config file for detection model')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint file for detection model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to point cloud file')
    parser.add_argument('--output_dir', type=str, default='./output', help='Base output directory')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    args = parser.parse_args()
    
    # Check if OpenPCDet is available
    if not ensure_openpcdet_available():
        return False
    
    # Make sure the checkpoint file exists
    if not os.path.exists(args.ckpt):
        print(f"Checkpoint file not found: {args.ckpt}")
        return False
    
    # Make sure the data file exists
    if not os.path.exists(args.data_path):
        print(f"Point cloud file not found: {args.data_path}")
        return False
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Use OpenPCDet's demo script directly
    success, model_output_dir = run_demo_script(
        args.openpcdet_dir, 
        args.cfg_file, 
        args.ckpt, 
        args.data_path, 
        args.output_dir
    )
    
    if not success:
        print("Detection failed")
        return False
    
    # Get the result file path
    pcd_filename = os.path.basename(args.data_path)
    pcd_name = os.path.splitext(pcd_filename)[0]
    result_path = os.path.join(model_output_dir, f'{pcd_name}.pkl')
    
    # Check if result file exists
    if not os.path.exists(result_path):
        print(f"Detection result file not found: {result_path}")
        result_path = None
    
    # Visualize results if requested
    if args.visualize and result_path:
        output_image = os.path.join(model_output_dir, f'{pcd_name}_detection.png')
        visualize_results(args.data_path, result_path, output_image)
    
    print(f"Detection complete! Results saved to {model_output_dir}")
    return True

if __name__ == "__main__":
    import time
    start_time = time.time()
    success = main()
    end_time = time.time()
    
    print(f"\nDetection {'succeeded' if success else 'failed'} in {end_time - start_time:.2f} seconds")
    sys.exit(0 if success else 1)
