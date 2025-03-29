import numpy as np
import open3d as o3d
import os
import glob
import time

def read_pcd_file(file_path):
    """
    Read a PCD file using Open3D.
    
    Args:
        file_path: Path to the PCD file
        
    Returns:
        PointCloud: Open3D PointCloud object
    """
    print(f"Reading {file_path}...")
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        print(f"  Loaded {len(pcd.points)} points")
        return pcd
    except Exception as e:
        print(f"  Error reading file: {e}")
        return None

def merge_point_clouds(pcds):
    """
    Merge multiple point clouds into one.
    
    Args:
        pcds: List of Open3D PointCloud objects
        
    Returns:
        PointCloud: Merged Open3D PointCloud object
    """
    print(f"Merging {len(pcds)} point clouds...")
    merged_pcd = o3d.geometry.PointCloud()
    
    for pcd in pcds:
        if pcd is not None:
            merged_pcd += pcd
    
    print(f"Merged point cloud has {len(merged_pcd.points)} points.")
    return merged_pcd

def visualize_point_cloud(pcd, window_name="Merged Point Cloud"):
    """
    Visualize a point cloud using Open3D's visualizer.
    
    Args:
        pcd: Open3D PointCloud object
        window_name: Name of the visualization window
    """
    print("Visualizing point cloud...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)
    vis.add_geometry(pcd)
    
    # Set some view control options
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    opt.point_size = 1.0
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()

def save_merged_point_cloud(pcd, output_path="merged_cie.pcd"):
    """
    Save the merged point cloud to a PCD file.
    
    Args:
        pcd: Open3D PointCloud object
        output_path: Path to save the merged PCD file
    """
    print(f"Saving merged point cloud to {output_path}...")
    o3d.io.write_point_cloud(output_path, pcd)
    print("Saved successfully.")

def main():
    """Main function to merge multiple PCD files."""
    # Directory containing PCD files
    pcd_dir = "./pcd_files"  # Replace with the actual directory
    
    # Find all PCD files
    pcd_files = glob.glob(os.path.join(pcd_dir, "*.pcd"))
    print(f"Found {len(pcd_files)} PCD files in {pcd_dir}")
    
    if not pcd_files:
        print("No PCD files found. Please check the directory.")
        return
    
    # Load all PCD files
    pcds = [read_pcd_file(file) for file in pcd_files]
    pcds = [pcd for pcd in pcds if pcd is not None]
    
    if not pcds:
        print("No valid PCD files loaded.")
        return
    
    # Merge point clouds
    merged_pcd = merge_point_clouds(pcds)
    
    # Save the merged point cloud
    save_merged_point_cloud(merged_pcd)
    
    # Visualize the merged point cloud
    visualize_point_cloud(merged_pcd)

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
