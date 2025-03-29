#!/usr/bin/env python
"""
CIE Point Cloud Object Detection - Main Entry Point
This script runs the complete pipeline for the ICDS Challenge
"""

import os
import argparse
import subprocess
import time

def run_command(command, description):
    """Run a command and print its output."""
    print(f"\n=== {description} ===")
    print(f"Running: {' '.join(command)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        elapsed_time = time.time() - start_time
        
        print(f"Output:\n{result.stdout}")
        if result.stderr:
            print(f"Errors/Warnings:\n{result.stderr}")
        
        print(f"Completed in {elapsed_time:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"Output:\n{e.stdout}")
        print(f"Error:\n{e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run the complete CIE Object Detection pipeline")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing CIE PCD files')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--use_openpcdet', action='store_true', help='Use OpenPCDet for detection (if available)')
    parser.add_argument('--cfg_file', type=str, help='Config file for OpenPCDet (required if use_openpcdet=True)')
    parser.add_argument('--ckpt', type=str, help='Checkpoint file for OpenPCDet (required if use_openpcdet=True)')
    parser.add_argument('--skip_merge', action='store_true', help='Skip merging step (use existing merged file)')
    parser.add_argument('--eps', type=float, default=0.15, help='DBSCAN epsilon parameter')
    parser.add_argument('--min_points', type=int, default=100, help='Minimum points for a cluster')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='Voxel size for downsampling')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if using OpenPCDet but missing required arguments
    if args.use_openpcdet and (not args.cfg_file or not args.ckpt):
        print("Error: When using OpenPCDet, both --cfg_file and --ckpt must be provided")
        return
    
    # Step 1: Merge PCD files
    merged_pcd_path = os.path.join(args.output_dir, 'CIE.pcd')
    if not args.skip_merge and not os.path.exists(merged_pcd_path):
        merge_command = [
            'python', 'scripts/merge_colab_script.py',
            '--input_dir', args.input_dir,
            '--output_path', merged_pcd_path
        ]
        
        if args.visualize:
            merge_command.append('--visualize')
            
        if not run_command(merge_command, "Merging PCD Files"):
            print("Merging failed. Exiting.")
            return
    else:
        print(f"\n=== Using existing merged file: {merged_pcd_path} ===")
    
    # Step 2: Run object detection
    detection_command = [
        'python', 'scripts/cie_detection.py',
        '--input_dir', args.input_dir,
        '--output_dir', args.output_dir,
        '--eps', str(args.eps),
        '--min_points', str(args.min_points),
        '--voxel_size', str(args.voxel_size)
    ]
    
    if args.use_openpcdet:
        detection_command.extend([
            '--cfg_file', args.cfg_file,
            '--ckpt', args.ckpt
        ])
    
    if not run_command(detection_command, "Running Object Detection"):
        print("Object detection failed. Exiting.")
        return
    
    print("\n=== ICDS Challenge Pipeline Completed Successfully! ===")
    print(f"Results saved to {args.output_dir}")
    
    # Remind about visualization if available
    if os.path.exists(os.path.join(args.output_dir, 'detection_results.png')):
        print(f"Detection visualization saved to {os.path.join(args.output_dir, 'detection_results.png')}")
    

if __name__ == "__main__":
    main()
