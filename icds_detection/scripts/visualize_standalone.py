import argparse
import open3d as o3d
import numpy as np
import json
import os
import sys

def visualize_detections(pcd_path, result_path):
    """Visualize detection results"""
    print(f"Visualizing point cloud at: {pcd_path}")
    print(f"Loading detection results from: {result_path}")
    
    # Check if point cloud exists
    if not os.path.exists(pcd_path):
        print(f"Error: Point cloud file not found at {pcd_path}")
        return
    
    # Load point cloud
    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
        print(f"Loaded point cloud with {len(pcd.points)} points")
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return
    
    # Load detection results if they exist
    objects = []
    try:
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                objects = json.load(f)
            print(f"Loaded {len(objects)} objects from detection results")
        else:
            print(f"Warning: No detection results found at {result_path}")
    except Exception as e:
        print(f"Error loading detection results: {e}")
        print("Continuing with visualization of point cloud only")
    
    # Define colors for different classes
    class_colors = {
        'chair': [1, 0, 0],       # Red
        'table': [0, 1, 0],       # Green
        'monitor': [0, 0, 1],     # Blue
        'screen': [1, 1, 0],      # Yellow
        'projector': [1, 0, 1],   # Magenta
        'tv': [0, 1, 1],          # Cyan
        'unknown': [0.7, 0.7, 0.7]  # Gray
    }
    
    # Create visualization elements
    geometries = []
    
    # Add point cloud with a light gray color
    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    
    # If point cloud has colors, use them, otherwise use light gray
    if pcd.has_colors():
        pcd_vis.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
    else:
        pcd_vis.paint_uniform_color([0.8, 0.8, 0.8])
    
    geometries.append(pcd_vis)
    
    # Add bounding boxes for each detected object
    if objects:
        for obj in objects:
            # Get label, defaulting to 'unknown' if not present
            label = obj.get('label', 'unknown')
            
            # Get color for label
            color = class_colors.get(label, class_colors['unknown'])
            
            # Create bounding box
            min_bound = [
                obj['bbox']['min_x'], 
                obj['bbox']['min_y'], 
                obj['bbox']['min_z']
            ]
            max_bound = [
                obj['bbox']['max_x'], 
                obj['bbox']['max_y'], 
                obj['bbox']['max_z']
            ]
            
            # Check for valid bounds before creating bbox
            if all(min_val <= max_val for min_val, max_val in zip(min_bound, max_bound)):
                bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
                bbox.color = color
                geometries.append(bbox)
                
                # Optionally add text label
                # Note: Open3D doesn't directly support text labels, 
                # but we could create a visualization for this
    
    # Print statistics
    print("\nDetection Statistics:")
    if objects:
        labels = [obj.get('label', 'unknown') for obj in objects]
        label_counts = {label: labels.count(label) for label in set(labels)}
        for label, count in label_counts.items():
            print(f"  {label}: {count}")
    else:
        print("  No objects detected")
    
    # Visualize
    if geometries:
        print("\nStarting visualization...")
        print("Controls: ")
        print("  Left-click + drag: Rotate")
        print("  Right-click + drag: Pan")
        print("  Mouse wheel: Zoom")
        print("  'r': Reset view")
        print("  'q': Exit")
        o3d.visualization.draw_geometries(geometries, window_name="CIE Point Cloud Detection Results")
    else:
        print("Error: No geometries to visualize")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Visualize Point Cloud Object Detection Results')
    parser.add_argument('--pcd_path', required=True, help='Path to point cloud file')
    parser.add_argument('--result_path', required=True, help='Path to detection results file')
    args = parser.parse_args()
    
    # Visualize detections
    visualize_detections(args.pcd_path, args.result_path)

if __name__ == "__main__":
    main()