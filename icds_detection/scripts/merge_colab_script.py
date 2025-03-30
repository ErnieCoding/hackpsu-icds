import os
import numpy as np
import open3d as o3d
import argparse
from tqdm import tqdm

def merge_pcd_files(input_dir, output_path, visualize=False):
    """
    Merge PCD files following the same logic as the provided Colab script.
    
    Args:
        input_dir: Root directory containing the PCD files
        output_path: Path to save the merged PCD file
        visualize: Whether to visualize the merged point cloud
    """
    print("Merging PCD files according to the Colab script logic...")
    
    # Define the structure based on the Colab script
    sections = {
        "ceilingtheater": range(3),
        "theater": range(7),
        "main": range(13),
        "secondary": range(11)
    }
    
    # Initialize combined point cloud
    CIE = o3d.geometry.PointCloud()
    
    # Process each section
    for section_name, file_range in sections.items():
        print(f"Processing {section_name} section...")
        section_cloud = o3d.geometry.PointCloud()
        
        # Loop through files in the section
        for i in file_range:
            file_path = os.path.join(input_dir, section_name, f"{section_name}_{i}.pcd")
            
            if os.path.exists(file_path):
                try:
                    pcd = o3d.io.read_point_cloud(file_path)
                    section_cloud += pcd
                    print(f"  Added {len(pcd.points)} points from {file_path}")
                except Exception as e:
                    print(f"  Error reading {file_path}: {e}")
            else:
                print(f"  Warning: File not found - {file_path}")
        
        # Save individual section (like in the Colab script)
        section_output = os.path.join(os.path.dirname(output_path), f"{section_name}.pcd")
        o3d.io.write_point_cloud(section_output, section_cloud, write_ascii=True)
        print(f"  Saved {section_name} point cloud with {len(section_cloud.points)} points")
        
        # Add to combined cloud
        CIE += section_cloud
    
    # Save combined point cloud
    o3d.io.write_point_cloud(output_path, CIE, write_ascii=True)
    print(f"Saved merged CIE point cloud with {len(CIE.points)} points to {output_path}")
    
    # Create downsampled version for visualization
    if visualize:
        print("Creating downsampled point cloud for visualization...")
        downpcd = CIE.voxel_down_sample(voxel_size=0.3)
        downsampled_path = os.path.join(os.path.dirname(output_path), "CIE_downsampled.pcd")
        o3d.io.write_point_cloud(downsampled_path, downpcd, write_ascii=True)
        print(f"Saved downsampled point cloud with {len(downpcd.points)} points")
        
        # Visualize
        print("Visualizing downsampled point cloud...")
        o3d.visualization.draw_geometries([downpcd])
    
    return CIE

def main():
    parser = argparse.ArgumentParser(description="Merge PCD files using the Colab script logic")
    parser.add_argument('--input_dir', type=str, required=True, help='Root directory containing the PCD files')
    parser.add_argument('--output_path', type=str, default='CIE.pcd', help='Output path for the merged PCD file')
    parser.add_argument('--visualize', action='store_true', help='Visualize the merged point cloud')
    args = parser.parse_args()
    
    merge_pcd_files(args.input_dir, args.output_path, args.visualize)

if __name__ == "__main__":
    main()
