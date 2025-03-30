import os
import numpy as np
import open3d as o3d
import torch
import argparse
from tqdm import tqdm
import time
import yaml
import sklearn.cluster as cluster
from sklearn.decomposition import PCA

# Import from OpenPCDet if available
try:
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils
    OPENPCDET_AVAILABLE = True
except ImportError:
    print("Warning: OpenPCDet not available. Running in standalone mode.")
    OPENPCDET_AVAILABLE = False

# Import the merging script
from merge_colab_script import merge_pcd_files

def preprocess_point_cloud(pcd, voxel_size=0.05):
    """
    Preprocess the point cloud by:
    1. Downsampling using voxel grid
    2. Estimating normals if not present
    3. Statistical outlier removal
    
    Args:
        pcd: Open3D PointCloud object
        voxel_size: Voxel size for downsampling
        
    Returns:
        PointCloud: Processed Open3D PointCloud object
    """
    print("Preprocessing point cloud...")
    
    # Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size)
    print(f"Downsampled to {len(pcd_down.points)} points")
    
    # Estimate normals if not present
    if not pcd_down.has_normals():
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        print("Normals estimated")
    
    # Statistical outlier removal
    pcd_clean, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"After outlier removal: {len(pcd_clean.points)} points")
    
    return pcd_clean

def segment_objects(pcd, eps=0.15, min_points=100):
    """
    Segment the point cloud into potential objects using DBSCAN clustering.
    
    Args:
        pcd: Open3D PointCloud object
        eps: DBSCAN epsilon parameter
        min_points: DBSCAN min_samples parameter
        
    Returns:
        list: List of point cloud segments (potential objects)
    """
    print("Segmenting objects...")
    
    # Convert to numpy array for DBSCAN
    points = np.asarray(pcd.points)
    
    # Run DBSCAN clustering
    db = cluster.DBSCAN(eps=eps, min_samples=min_points, n_jobs=-1).fit(points)
    labels = db.labels_
    
    # Number of clusters (excluding noise points with label -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {n_clusters} potential objects")
    
    # Create a list of point cloud segments
    segments = []
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) < min_points:  # Skip very small clusters
            continue
            
        # Create a new point cloud for this segment
        segment = o3d.geometry.PointCloud()
        segment.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[cluster_indices])
        if pcd.has_colors():
            segment.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[cluster_indices])
        if pcd.has_normals():
            segment.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[cluster_indices])
        
        segments.append(segment)
    
    return segments

def classify_objects(segments):
    """
    Classify segments into object categories based on simple geometric features.
    
    Args:
        segments: List of Open3D PointCloud objects
        
    Returns:
        list: List of (segment, class_name, confidence) tuples
    """
    print("Classifying objects...")
    
    results = []
    
    # Define object classes with their expected geometric properties
    class_properties = {
        'Chair': {'height_range': (0.4, 1.2), 'volume_range': (0.05, 0.5), 'aspect_ratio_range': (0.5, 2.0)},
        'Table': {'height_range': (0.5, 1.1), 'volume_range': (0.3, 5.0), 'aspect_ratio_range': (0.5, 2.0)},
        'Monitor': {'height_range': (0.2, 0.6), 'volume_range': (0.01, 0.3), 'aspect_ratio_range': (1.5, 5.0)},
        'Projection_Screen': {'height_range': (0.8, 2.5), 'volume_range': (0.3, 10.0), 'aspect_ratio_range': (1.5, 10.0)},
        'Projector': {'height_range': (0.1, 0.5), 'volume_range': (0.01, 0.1), 'aspect_ratio_range': (0.5, 2.0)},
        'TV': {'height_range': (0.3, 1.0), 'volume_range': (0.05, 0.8), 'aspect_ratio_range': (1.5, 5.0)},
        'Computer': {'height_range': (0.2, 0.7), 'volume_range': (0.01, 0.3), 'aspect_ratio_range': (0.5, 2.0)}
    }
    
    for segment in tqdm(segments, desc="Classifying"):
        # Get segment properties
        points = np.asarray(segment.points)
        
        # Get oriented bounding box
        obb = segment.get_oriented_bounding_box()
        dimensions = obb.extent
        
        # Calculate geometric features
        height = dimensions[2]  # Assuming Z is up
        volume = np.prod(dimensions)
        aspect_ratio = max(dimensions[0], dimensions[1]) / (min(dimensions[0], dimensions[1]) + 1e-6)
        
        # Find the best matching class
        best_class = 'Misc'
        best_score = 0.0
        
        for class_name, props in class_properties.items():
            # Calculate match score based on how well the segment matches the expected properties
            height_match = 1.0 if props['height_range'][0] <= height <= props['height_range'][1] else 0.0
            volume_match = 1.0 if props['volume_range'][0] <= volume <= props['volume_range'][1] else 0.0
            ar_match = 1.0 if props['aspect_ratio_range'][0] <= aspect_ratio <= props['aspect_ratio_range'][1] else 0.0
            
            score = (height_match + volume_match + ar_match) / 3.0
            
            if score > best_score:
                best_score = score
                best_class = class_name
        
        results.append((segment, best_class, best_score))
    
    return results

def visualize_results(pcd, classified_segments, output_path=None):
    """
    Visualize the original point cloud and the classified objects.
    
    Args:
        pcd: Original Open3D PointCloud object
        classified_segments: List of (segment, class_name, confidence) tuples
        output_path: Path to save the visualization image
    """
    print("Visualizing results...")
    
    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="CIE Object Detection Results", width=1280, height=720)
    
    # Add the original point cloud with low opacity
    pcd_vis = o3d.geometry.PointCloud(pcd)
    pcd_vis.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray
    vis.add_geometry(pcd_vis)
    
    # Color palette for different object classes
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
    
    # Add each segment with its class-specific color
    for segment, class_name, confidence in classified_segments:
        # Skip low confidence predictions
        if confidence < 0.3:
            continue
            
        # Create a copy of the segment for visualization
        segment_vis = o3d.geometry.PointCloud(segment)
        
        # Color based on class
        if class_name in colors:
            segment_vis.paint_uniform_color(colors[class_name])
        else:
            segment_vis.paint_uniform_color(colors['Misc'])
        
        # Add to visualizer
        vis.add_geometry(segment_vis)
        
        # Add bounding box
        bbox = segment.get_oriented_bounding_box()
        bbox.color = colors[class_name] if class_name in colors else colors['Misc']
        vis.add_geometry(bbox)
    
    # Set some view control options
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    opt.point_size = 2.0
    
    # Run the visualizer
    vis.run()
    
    # Capture and save an image if requested
    if output_path:
        vis.capture_screen_image(output_path)
        print(f"Visualization saved to {output_path}")
    
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description="CIE Point Cloud Object Detection")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing CIE PCD files')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--cfg_file', type=str, help='Config file for OpenPCDet (if available)')
    parser.add_argument('--ckpt', type=str, help='Checkpoint file for OpenPCDet (if available)')
    parser.add_argument('--eps', type=float, default=0.15, help='DBSCAN epsilon parameter')
    parser.add_argument('--min_points', type=int, default=100, help='Minimum points for a cluster')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='Voxel size for downsampling')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Merge PCD files using the Colab script logic
    merged_pcd_path = os.path.join(args.output_dir, 'CIE.pcd')
    if not os.path.exists(merged_pcd_path):
        pcd = merge_pcd_files(args.input_dir, merged_pcd_path)
    else:
        print(f"Using existing merged point cloud: {merged_pcd_path}")
        pcd = o3d.io.read_point_cloud(merged_pcd_path)
    
    # Step 2: Preprocess point cloud
    processed_pcd = preprocess_point_cloud(pcd, voxel_size=args.voxel_size)
    processed_pcd_path = os.path.join(args.output_dir, 'CIE_processed.pcd')
    o3d.io.write_point_cloud(processed_pcd_path, processed_pcd)
    
    # Use OpenPCDet if available and config is provided
    if OPENPCDET_AVAILABLE and args.cfg_file and args.ckpt:
        print("Using OpenPCDet for object detection...")
        # Load config
        cfg_from_yaml_file(args.cfg_file, cfg)
        
        # Build model
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=None)
        
        # Load checkpoint
        model.load_params_from_file(filename=args.ckpt, logger=common_utils.create_logger())
        model.cuda()
        model.eval()
        
        # Process point cloud for OpenPCDet
        # Implementation depends on specific OpenPCDet requirements
        # This is a placeholder for the actual implementation
        print("OpenPCDet implementation to be completed based on specific requirements")
        
    else:
        print("Using standalone object detection pipeline...")
        # Step 3: Segment objects
        segments = segment_objects(processed_pcd, eps=args.eps, min_points=args.min_points)
        
        # Step 4: Classify objects
        classified_segments = classify_objects(segments)
        
        # Step 5: Visualize results
        vis_output_path = os.path.join(args.output_dir, 'detection_results.png')
        visualize_results(processed_pcd, classified_segments, vis_output_path)
        
        # Write results to a file
        results_file = os.path.join(args.output_dir, 'detection_results.txt')
        with open(results_file, 'w') as f:
            f.write("CIE Object Detection Results\n")
            f.write("==========================\n\n")
            
            # Group by class
            class_counts = {}
            for _, class_name, confidence in classified_segments:
                if confidence >= 0.3:  # Only count high confidence detections
                    if class_name not in class_counts:
                        class_counts[class_name] = 0
                    class_counts[class_name] += 1
            
            f.write("Object counts:\n")
            for class_name, count in class_counts.items():
                f.write(f"{class_name}: {count}\n")
            
            f.write("\nDetailed results:\n")
            for i, (segment, class_name, confidence) in enumerate(classified_segments):
                if confidence >= 0.3:  # Only include high confidence detections
                    bbox = segment.get_oriented_bounding_box()
                    center = bbox.center
                    dimensions = bbox.extent
                    
                    f.write(f"Object {i+1}:\n")
                    f.write(f"  Class: {class_name}\n")
                    f.write(f"  Confidence: {confidence:.2f}\n")
                    f.write(f"  Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})\n")
                    f.write(f"  Dimensions: {dimensions[0]:.2f} x {dimensions[1]:.2f} x {dimensions[2]:.2f}\n")
                    f.write(f"  Points: {len(segment.points)}\n\n")
        
        print(f"Results written to {results_file}")
    
    print("Object detection completed successfully!")

if __name__ == "__main__":
    main()
