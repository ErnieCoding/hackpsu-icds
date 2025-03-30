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
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"Output:\n{e.stdout}")
        print(f"Error:\n{e.stderr}")
        return False, e.stderr

def main():
    parser = argparse.ArgumentParser(description="Run the complete CIE Object Detection pipeline")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing CIE PCD files')
    parser.add_argument('--output_dir', type=str, default='./output', help='Base output directory')
    parser.add_argument('--use_openpcdet', action='store_true', help='Use OpenPCDet for detection (if available)')
    parser.add_argument('--cfg_file', type=str, help='Config file for OpenPCDet (required if use_openpcdet=True)')
    parser.add_argument('--ckpt', type=str, help='Checkpoint file for OpenPCDet (required if use_openpcdet=True)')
    parser.add_argument('--skip_merge', action='store_true', help='Skip merging step (use existing merged file)')
    parser.add_argument('--eps', type=float, default=0.15, help='DBSCAN epsilon parameter')
    parser.add_argument('--min_points', type=int, default=100, help='Minimum points for a cluster')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='Voxel size for downsampling')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--run_all_models', action='store_true', help='Run all available OpenPCDet models')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if using OpenPCDet but missing required arguments
    if args.use_openpcdet and not args.run_all_models and (not args.cfg_file or not args.ckpt):
        print("Error: When using OpenPCDet, both --cfg_file and --ckpt must be provided (or use --run_all_models)")
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
            
        success, _ = run_command(merge_command, "Merging PCD Files")
        if not success:
            print("Merging failed. Exiting.")
            return
    else:
        print(f"\n=== Using existing merged file: {merged_pcd_path} ===")
    
    # Step 2: Run standalone object detection
    standalone_command = [
        'python', 'scripts/cie_detection.py',
        '--input_dir', args.input_dir,
        '--output_dir', args.output_dir,
        '--eps', str(args.eps),
        '--min_points', str(args.min_points),
        '--voxel_size', str(args.voxel_size)
    ]
    
    success, _ = run_command(standalone_command, "Running Standalone Object Detection")
    if not success:
        print("Standalone object detection failed.")
    else:
        print(f"Standalone detection results saved to {os.path.join(args.output_dir, 'standalone_detection')}")
    
    # Step 3: Run OpenPCDet detection (if requested)
    if args.use_openpcdet:
        # If run_all_models is specified, try to find all available models
        if args.run_all_models:
            print("\n=== Running all available OpenPCDet models ===")
            # Define available models and their checkpoints
            # You would need to provide the paths to these checkpoints
            models = {
                "pointpillar": {"cfg": "tools/cfgs/cie_models/cie_pointpillar.yaml", 
                               "ckpt": "checkpoints/pointpillar_7728.pth"},
                "second": {"cfg": "tools/cfgs/cie_models/cie_second.yaml", 
                          "ckpt": "checkpoints/second_7862.pth"},
                "pv_rcnn": {"cfg": "tools/cfgs/cie_models/cie_pv_rcnn.yaml", 
                           "ckpt": "checkpoints/pv_rcnn_8369.pth"}
            }
            
            # Check which models have available checkpoints
            available_models = {}
            for model_name, model_info in models.items():
                if os.path.exists(model_info["ckpt"]):
                    available_models[model_name] = model_info
            
            if not available_models:
                print("No available models found with checkpoints. Please download models first.")
            else:
                print(f"Found {len(available_models)} available models: {', '.join(available_models.keys())}")
                
                # Run each available model
                for model_name, model_info in available_models.items():
                    print(f"\n=== Running OpenPCDet detection with {model_name} model ===")
                    openpcdet_command = [
                        'python', 'scripts/run_openpcdet_detection.py',
                        '--openpcdet_dir', './OpenPCDet',
                        '--cfg_file', model_info["cfg"],
                        '--ckpt', model_info["ckpt"],
                        '--data_path', merged_pcd_path,
                        '--output_dir', args.output_dir
                    ]
                    
                    if args.visualize:
                        openpcdet_command.append('--visualize')
                        
                    success, output = run_command(openpcdet_command, f"Running OpenPCDet Detection with {model_name}")
                    if not success:
                        print(f"OpenPCDet detection with {model_name} failed.")
                    else:
                        # Extract the output directory from the command output
                        output_dir = None
                        for line in output.split('\n'):
                            if "Results saved to" in line:
                                output_dir = line.split("Results saved to")[-1].strip()
                                break
                        
                        if output_dir:
                            print(f"{model_name} detection results saved to {output_dir}")
                        else:
                            print(f"{model_name} detection completed, but output directory not found in output.")
        else:
            # Run with the specified model
            openpcdet_command = [
                'python', 'scripts/run_openpcdet_detection.py',
                '--openpcdet_dir', './OpenPCDet',
                '--cfg_file', args.cfg_file,
                '--ckpt', args.ckpt,
                '--data_path', merged_pcd_path,
                '--output_dir', args.output_dir
            ]
            
            if args.visualize:
                openpcdet_command.append('--visualize')
                
            success, output = run_command(openpcdet_command, "Running OpenPCDet Detection")
            if not success:
                print("OpenPCDet detection failed.")
            else:
                # Extract the output directory from the command output
                output_dir = None
                for line in output.split('\n'):
                    if "Results saved to" in line:
                        output_dir = line.split("Results saved to")[-1].strip()
                        break
                
                if output_dir:
                    print(f"OpenPCDet detection results saved to {output_dir}")
                else:
                    print("OpenPCDet detection completed, but output directory not found in output.")
    
    print("\n=== ICDS Challenge Pipeline Completed Successfully! ===")
    print(f"All results saved under {args.output_dir}")
    
    # Summary of output locations
    print("\nOutput Locations:")
    print(f"- Merged point cloud: {merged_pcd_path}")
    print(f"- Standalone detection results: {os.path.join(args.output_dir, 'standalone_detection')}")
    
    if args.use_openpcdet:
        print("- OpenPCDet detection results:")
        if args.run_all_models and 'available_models' in locals():
            for model_name in available_models.keys():
                print(f"  - {model_name}: {os.path.join(args.output_dir, f'openpcdet_{model_name}')}")
        else:
            model_name = os.path.splitext(os.path.basename(args.cfg_file))[0]
            print(f"  - {model_name}: {os.path.join(args.output_dir, f'openpcdet_{model_name}')}")
    

if __name__ == "__main__":
    main()
