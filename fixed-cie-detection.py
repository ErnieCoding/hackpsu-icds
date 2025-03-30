#!/usr/bin/env python
"""
Improved CIE Point Cloud Object Detection

This script:
1. Loads and preprocesses the point cloud data
2. Segments the point cloud to identify potential objects
3. Classifies each segment based on geometric features and colors
4. Applies contextual rules (e.g., monitors on tables, TVs on walls)
5. Visualizes and saves the results
"""

import os
import sys
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn import cluster
from tqdm import tqdm
import time
from collections import defaultdict

def merge_pcd_files(input_dir, output_path):
    """
    Merge multiple point cloud files into one.
    
    Args:
        input_dir: Directory containing PCD files
        output_path: Path to save the merged point cloud
        
    Returns:
        PointCloud: Merged Open3D PointCloud object
    """
    print(f"Merging PCD files from {input_dir}...")
    
    # List all PCD files in the directory
    pcd_files = [f for f in os.listdir(input_dir) if f.endswith('.pcd')]
    if not pcd_files:
        raise ValueError(f"No PCD files found in {input_dir}")
    
    print(f"Found {len(pcd_files)} PCD files")
    
    # Initialize with the first point cloud
    merged_pcd = o3d.io.read_point_cloud(os.path.join(input_dir, pcd_files[0]))
    print(f"Loaded {pcd_files[0]}, points: {len(merged_pcd.points)}")
    
    # Add the rest of the point clouds
    for i in range(1, len(pcd_files)):
        pcd_file = pcd_files[i]
        print(f"Loading {pcd_file}...")
        pcd = o3d.io.read_point_cloud(os.path.join(input_dir, pcd_file))
        print(f"  Points: {len(pcd.points)}")
        merged_pcd += pcd
    
    print(f"Total points after merging: {len(merged_pcd.points)}")
    
    # Save the merged point cloud
    o3d.io.write_point_cloud(output_path, merged_pcd)
    print(f"Merged point cloud saved to {output_path}")
    
    return merged_pcd

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

def segment_objects(pcd, eps=0.1, min_points=50):
    """
    Segment point cloud into potential objects using DBSCAN clustering.
    
    Args:
        pcd: Open3D PointCloud object
        eps: DBSCAN epsilon parameter
        min_points: DBSCAN min_samples parameter
        
    Returns:
        list: List of point cloud segments (potential objects)
    """
    print("Segmenting objects...")
    print(f"Input point cloud has {len(pcd.points)} points")
    
    # Convert to numpy array for DBSCAN
    points = np.asarray(pcd.points)
    print(f"Running DBSCAN clustering with eps={eps}, min_points={min_points}...")
    
    # Run DBSCAN clustering
    db = cluster.DBSCAN(eps=eps, min_samples=min_points).fit(points)
    labels = db.labels_
    
    # Number of clusters (excluding noise points with label -1)
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    print(f"DBSCAN found {n_clusters} potential clusters")
    print(f"Unique labels: {unique_labels}")
    
    # Create a list of point cloud segments
    segments = []
    
    # Process each DBSCAN cluster
    for label in unique_labels:
        if label == -1:  # Skip noise
            continue
            
        cluster_indices = np.where(labels == label)[0]
        print(f"Cluster {label}: {len(cluster_indices)} points")
        
        if len(cluster_indices) < min_points:
            print(f"  Skipping cluster {label}: too few points")
            continue
            
        # Create a new point cloud for this segment
        segment = o3d.geometry.PointCloud()
        segment.points = o3d.utility.Vector3dVector(points[cluster_indices])
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            segment.colors = o3d.utility.Vector3dVector(colors[cluster_indices])
        
        # Get the axis-aligned bounding box
        bbox = segment.get_axis_aligned_bounding_box()
        min_bound = bbox.min_bound
        max_bound = bbox.max_bound
        
        # Calculate dimensions
        width = max_bound[0] - min_bound[0]   # X-axis
        length = max_bound[1] - min_bound[1]  # Y-axis
        height = max_bound[2] - min_bound[2]  # Z-axis
        volume = width * length * height
        
        print(f"  Dimensions: {width:.2f}m x {length:.2f}m x {height:.2f}m, Volume: {volume:.2f}m³")
        
        # Filter out clusters that are too large or too small
        MAX_DIMENSION = 5.0  # meters
        MAX_VOLUME = 10.0     # cubic meters
        MIN_VOLUME = 0.001    # cubic meters
        
        if width > MAX_DIMENSION or length > MAX_DIMENSION or height > MAX_DIMENSION:
            print(f"  Skipping cluster {label}: dimension too large")
            continue
            
        if volume > MAX_VOLUME:
            print(f"  Skipping cluster {label}: volume too large")
            continue
            
        if volume < MIN_VOLUME:
            print(f"  Skipping cluster {label}: volume too small")
            continue
        
        print(f"  Adding cluster {label} to segments")
        segments.append(segment)
    
    print(f"Final number of segments: {len(segments)}")
    return segments

def calculate_average_color(segment):
    """
    Calculate the average color of a segment.
    
    Args:
        segment: Open3D PointCloud segment
        
    Returns:
        numpy.ndarray: Average RGB color
    """
    if segment.has_colors():
        colors = np.asarray(segment.colors)
        return np.mean(colors, axis=0)
    else:
        return np.array([0.5, 0.5, 0.5])  # Default gray if no colors

def color_distance(color1, color2):
    """
    Calculate Euclidean distance between two RGB colors.
    
    Args:
        color1: First RGB color array
        color2: Second RGB color array
        
    Returns:
        float: Distance between colors
    """
    return np.linalg.norm(color1 - color2)

def get_height_from_floor(segment):
    """
    Calculate the height of an object from the floor.
    
    Args:
        segment: Open3D PointCloud segment
        
    Returns:
        float: Height from floor (minimum z value)
    """
    points = np.asarray(segment.points)
    return np.min(points[:, 2])

def is_on_wall(segment, room_dims):
    """
    Check if an object is likely attached to a wall.
    
    Args:
        segment: Open3D PointCloud segment
        room_dims: Tuple of room dimensions (width, length, height)
        
    Returns:
        bool: True if the object is likely on a wall
    """
    bbox = segment.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound
    
    # Check proximity to room boundaries
    room_width, room_length, _ = room_dims
    tolerance = 0.3  # meters
    
    # Check if the object is near any of the four walls
    near_x_min = min_bound[0] < tolerance
    near_x_max = max_bound[0] > (room_width - tolerance)
    near_y_min = min_bound[1] < tolerance
    near_y_max = max_bound[1] > (room_length - tolerance)
    
    return near_x_min or near_x_max or near_y_min or near_y_max

def classify_objects(segments, room_dims=(9.1, 10.7, 3.35)):
    """
    Classify segments into object categories based on geometric features, colors,
    and contextual information.
    
    Args:
        segments: List of Open3D PointCloud objects (segments)
        room_dims: Approximate room dimensions in meters (width, length, height)
        
    Returns:
        list: List of (segment, class_name, confidence, metadata) tuples
    """
    print("Classifying objects...")
    
    # Convert inch measurements to meters
    inches_to_meters = 0.0254
    
    # Define object dimensions in meters
    # Chair: 20" x 20" x 41-53" (0.51m x 0.51m x 1.04-1.35m)
    CHAIR_WIDTH_MIN, CHAIR_WIDTH_MAX = 0.4, 0.6
    CHAIR_LENGTH_MIN, CHAIR_LENGTH_MAX = 0.4, 0.6
    CHAIR_HEIGHT_MIN, CHAIR_HEIGHT_MAX = 1.0, 1.4
    
    # Table: 24" x 24-60" x 29-30" (0.61m x 0.61-1.52m x 0.74-0.76m)
    TABLE_WIDTH_MIN, TABLE_WIDTH_MAX = 0.6, 1.6
    TABLE_LENGTH_MIN, TABLE_LENGTH_MAX = 0.6, 1.6
    TABLE_HEIGHT_MIN, TABLE_HEIGHT_MAX = 0.7, 0.8
    
    # Monitor: 22" x <1" x 12" (0.56m x 0.025m x 0.3m) - corrected orientation
    MONITOR_WIDTH_MIN, MONITOR_WIDTH_MAX = 0.45, 0.65
    MONITOR_DEPTH_MIN, MONITOR_DEPTH_MAX = 0.01, 0.1
    MONITOR_HEIGHT_MIN, MONITOR_HEIGHT_MAX = 0.25, 0.35
    
    # Screen (projection): 8' x thin x 4'5" (2.44m x thin x 1.35m)
    SCREEN_WIDTH_MIN, SCREEN_WIDTH_MAX = 2.0, 3.0
    SCREEN_DEPTH_MIN, SCREEN_DEPTH_MAX = 0.01, 0.2
    SCREEN_HEIGHT_MIN, SCREEN_HEIGHT_MAX = 1.2, 1.5
    
    # Projector: 17" x 12" x 5" (0.43m x 0.3m x 0.13m)
    PROJECTOR_WIDTH_MIN, PROJECTOR_WIDTH_MAX = 0.35, 0.5
    PROJECTOR_LENGTH_MIN, PROJECTOR_LENGTH_MAX = 0.25, 0.35
    PROJECTOR_HEIGHT_MIN, PROJECTOR_HEIGHT_MAX = 0.1, 0.15
    
    # TV: 40"-80" diagonally, assume rectangular shape
    TV_WIDTH_MIN, TV_WIDTH_MAX = 0.8, 1.8
    TV_DEPTH_MIN, TV_DEPTH_MAX = 0.05, 0.15
    TV_HEIGHT_MIN, TV_HEIGHT_MAX = 0.45, 1.1
    
    # Define object colors (RGB normalized to [0,1])
    CHAIR_COLOR = np.array([47/255, 49/255, 48/255])
    TABLE_COLOR = np.array([215/255, 178/255, 119/255])
    MONITOR_COLOR = np.array([27/255, 36/255, 51/255])
    PROJECTOR_COLOR = np.array([92/255, 92/255, 94/255])
    TV_COLOR = np.array([0/255, 163/255, 251/255])
    SCREEN_COLOR = np.array([207/255, 211/255, 210/255])
    
    # Color distance thresholds
    COLOR_THRESHOLD = 0.3  # Lower = stricter color matching
    
    # Store spatial relationships for post-processing
    tables = []
    monitors = []
    tvs = []
    all_objects = []
    
    results = []
    
    for segment in tqdm(segments, desc="Classifying"):
        # Get segment properties
        bbox = segment.get_axis_aligned_bounding_box()
        min_bound = bbox.min_bound
        max_bound = bbox.max_bound
        
        # Calculate dimensions
        width = max_bound[0] - min_bound[0]   # X-axis
        length = max_bound[1] - min_bound[1]  # Y-axis
        height = max_bound[2] - min_bound[2]  # Z-axis
        
        # Sort dimensions to handle orientation-independent classification
        dims = sorted([width, length], reverse=True)
        largest_dim, second_dim = dims[0], dims[1]
        
        # Calculate volume and aspect ratios
        volume = width * length * height
        flatness_ratio = height / (largest_dim + 1e-6)  # How flat is it? (height/width)
        elongation_ratio = largest_dim / (second_dim + 1e-6)  # How elongated? (length/width)
        
        # Skip objects that are too large (likely walls/floors/ceilings)
        if largest_dim > 5.0 or volume > 10.0:
            continue
        
        # Skip objects that are too small (likely noise)
        if volume < 0.01:  # 10 cubic cm
            continue
        
        # Get average color
        avg_color = calculate_average_color(segment)
        
        # Calculate color distances
        chair_color_dist = color_distance(avg_color, CHAIR_COLOR)
        table_color_dist = color_distance(avg_color, TABLE_COLOR)
        monitor_color_dist = color_distance(avg_color, MONITOR_COLOR)
        projector_color_dist = color_distance(avg_color, PROJECTOR_COLOR)
        tv_color_dist = color_distance(avg_color, TV_COLOR)
        screen_color_dist = color_distance(avg_color, SCREEN_COLOR)
        
        # Check if object is on wall
        on_wall = is_on_wall(segment, room_dims)
        
        # Height from floor
        height_from_floor = get_height_from_floor(segment)
        
        # Initialize class, confidence, and metadata
        class_name = "Misc"
        confidence = 0.5
        metadata = {
            "dimensions": (width, length, height),
            "volume": volume,
            "avg_color": avg_color,
            "on_wall": on_wall,
            "height_from_floor": height_from_floor
        }
        
        # Table classification
        if (TABLE_WIDTH_MIN <= largest_dim <= TABLE_WIDTH_MAX and 
            TABLE_LENGTH_MIN <= second_dim <= TABLE_LENGTH_MAX and 
            TABLE_HEIGHT_MIN <= height <= TABLE_HEIGHT_MAX and
            flatness_ratio < 0.6 and  # Tables are flat
            table_color_dist < COLOR_THRESHOLD):
            class_name = "Table"
            confidence = 0.8
            tables.append((segment, len(all_objects)))  # Store for spatial relationships
        
        # Chair classification
        elif (CHAIR_WIDTH_MIN <= largest_dim <= CHAIR_WIDTH_MAX and 
              CHAIR_LENGTH_MIN <= second_dim <= CHAIR_LENGTH_MAX and 
              CHAIR_HEIGHT_MIN <= height <= CHAIR_HEIGHT_MAX and
              chair_color_dist < COLOR_THRESHOLD):
            class_name = "Chair"
            confidence = 0.8
        
        # Monitor classification
        elif (MONITOR_WIDTH_MIN <= largest_dim <= MONITOR_WIDTH_MAX and 
              MONITOR_DEPTH_MIN <= second_dim <= MONITOR_DEPTH_MAX and 
              MONITOR_HEIGHT_MIN <= height <= MONITOR_HEIGHT_MAX and
              monitor_color_dist < COLOR_THRESHOLD):
            class_name = "Monitor"
            confidence = 0.75
            monitors.append((segment, len(all_objects)))  # Store for spatial relationships
        
        # Projection screen classification - only one in theatre
        elif (SCREEN_WIDTH_MIN <= largest_dim <= SCREEN_WIDTH_MAX and 
              SCREEN_DEPTH_MIN <= second_dim <= SCREEN_DEPTH_MAX and 
              SCREEN_HEIGHT_MIN <= height <= SCREEN_HEIGHT_MAX and
              on_wall and  # Must be on wall
              screen_color_dist < COLOR_THRESHOLD):
            class_name = "Projection_Screen"
            confidence = 0.9
        
        # Projector classification
        elif (PROJECTOR_WIDTH_MIN <= largest_dim <= PROJECTOR_WIDTH_MAX and 
              PROJECTOR_LENGTH_MIN <= second_dim <= PROJECTOR_LENGTH_MAX and 
              PROJECTOR_HEIGHT_MIN <= height <= PROJECTOR_HEIGHT_MAX and
              projector_color_dist < COLOR_THRESHOLD):
            class_name = "Projector"
            confidence = 0.8
        
        # TV classification
        elif (TV_WIDTH_MIN <= largest_dim <= TV_WIDTH_MAX and 
              TV_DEPTH_MIN <= second_dim <= TV_DEPTH_MAX and 
              TV_HEIGHT_MIN <= height <= TV_HEIGHT_MAX and
              on_wall and  # TVs must be on walls
              tv_color_dist < COLOR_THRESHOLD):
            class_name = "TV"
            confidence = 0.85
            tvs.append((segment, len(all_objects)))  # Store for spatial relationships
        
        # Calculate center position
        center = (min_bound + max_bound) / 2
        metadata["center"] = center
        
        # Print debug info for larger objects
        if volume > 0.5:
            print(f"Large object: dims={width:.2f}x{length:.2f}x{height:.2f}m, "
                  f"vol={volume:.2f}m³, center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}), "
                  f"class={class_name}, color_dist={min([chair_color_dist, table_color_dist, monitor_color_dist, projector_color_dist, tv_color_dist, screen_color_dist]):.2f}")
        
        results.append((segment, class_name, confidence, metadata))
        all_objects.append((segment, class_name, confidence, metadata))
    
    # Post-processing: Apply contextual rules
    
    # Rule 1: Monitor must be on a table
    for monitor_idx, (monitor_segment, obj_idx) in enumerate(monitors):
        monitor_center = all_objects[obj_idx][3]["center"]
        monitor_bbox = monitor_segment.get_axis_aligned_bounding_box()
        monitor_min = monitor_bbox.min_bound
        
        on_table = False
        for table_segment, table_idx in tables:
            table_center = all_objects[table_idx][3]["center"]
            table_bbox = table_segment.get_axis_aligned_bounding_box()
            table_max = table_bbox.max_bound
            
            # Check if monitor is positioned above table
            if (abs(monitor_center[0] - table_center[0]) < 0.5 and
                abs(monitor_center[1] - table_center[1]) < 0.5 and
                abs(monitor_min[2] - table_max[2]) < 0.3):
                on_table = True
                break
        
        # If monitor is not on a table, reduce confidence
        if not on_table and obj_idx < len(results):
            segment, class_name, confidence, metadata = results[obj_idx]
            if class_name == "Monitor":
                results[obj_idx] = (segment, class_name, confidence * 0.6, metadata)
    
    # Rule 2: Verify TVs are actually on walls
    for tv_idx, (tv_segment, obj_idx) in enumerate(tvs):
        if obj_idx < len(results):
            segment, class_name, confidence, metadata = results[obj_idx]
            if class_name == "TV" and not metadata["on_wall"]:
                results[obj_idx] = (segment, class_name, confidence * 0.5, metadata)
    
    # Rule 3: Limit to one projection screen
    screen_confidences = []
    for idx, (_, class_name, confidence, _) in enumerate(results):
        if class_name == "Projection_Screen":
            screen_confidences.append((idx, confidence))
    
    # Keep only the highest confidence screen
    if len(screen_confidences) > 1:
        # Sort by confidence (descending)
        screen_confidences.sort(key=lambda x: x[1], reverse=True)
        
        # Reduce confidence of all but the highest confidence screen
        for idx, _ in screen_confidences[1:]:
            segment, class_name, _, metadata = results[idx]
            results[idx] = (segment, "Misc", 0.4, metadata)
    
    # Count objects by class
    class_counts = {}
    for _, class_name, confidence, _ in results:
        if confidence >= 0.5:  # Only count medium-high confidence detections
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
    
    print("Object counts by class:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    
    # Filter to only return segment, class, confidence (without metadata) for compatibility
    final_results = [(segment, class_name, confidence) for segment, class_name, confidence, _ in results]
    return final_results

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
    
    # Color palette for different object classes (using specified colors)
    colors = {
        'Chair': [47/255, 49/255, 48/255],       # Specified chair color
        'Table': [215/255, 178/255, 119/255],    # Specified table color
        'Monitor': [27/255, 36/255, 51/255],     # Specified monitor color
        'Projection_Screen': [207/255, 211/255, 210/255],  # Specified screen color
        'Projector': [92/255, 92/255, 94/255],   # Specified projector color
        'TV': [0/255, 163/255, 251/255],         # Specified TV color
        'Misc': [0.5, 0.5, 0.5]                  # Gray
    }
    
    # Add each segment with its class-specific color
    for segment, class_name, confidence in classified_segments:
        # Skip low confidence predictions
        if confidence < 0.5:
            continue
            
        # Create a copy of the segment for visualization
        segment_vis = o3d.geometry.PointCloud()
        segment_vis.points = o3d.utility.Vector3dVector(np.asarray(segment.points))
        if segment.has_colors():
            segment_vis.colors = o3d.utility.Vector3dVector(np.asarray(segment.colors))
        
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
    
    # Capture image if needed
    if output_path:
        # Update the renderer before capturing
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(output_path)
        print(f"Visualization saved to {output_path}")
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description="CIE Point Cloud Object Detection")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing CIE PCD files')
    parser.add_argument('--output_dir', type=str, default='./output', help='Base output directory')
    parser.add_argument('--eps', type=float, default=0.1, help='DBSCAN epsilon parameter')
    parser.add_argument('--min_points', type=int, default=50, help='Minimum points for a cluster')
    parser.add_argument('--voxel_size', type=float, default=0.04, help='Voxel size for downsampling')
    args = parser.parse_args()
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create standalone output directory
    standalone_output_dir = os.path.join(args.output_dir, "standalone_detection")
    os.makedirs(standalone_output_dir, exist_ok=True)
    
    # Step 1: Merge PCD files
    merged_pcd_path = os.path.join(args.output_dir, 'CIE.pcd')
    if not os.path.exists(merged_pcd_path):
        pcd = merge_pcd_files(args.input_dir, merged_pcd_path)
    else:
        print(f"Using existing merged point cloud: {merged_pcd_path}")
        pcd = o3d.io.read_point_cloud(merged_pcd_path)
    
    # Step 2: Preprocess point cloud
    processed_pcd = preprocess_point_cloud(pcd, voxel_size=args.voxel_size)
    processed_pcd_path = os.path.join(standalone_output_dir, 'CIE_processed.pcd')
    o3d.io.write_point_cloud(processed_pcd_path, processed_pcd)
    
    # Step 3: Segment objects
    segments = segment_objects(processed_pcd, eps=args.eps, min_points=args.min_points)
    
    # Step 4: Classify objects
    classified_segments = classify_objects(segments)
    
    # Step 5: Visualize results
    vis_output_path = os.path.join(standalone_output_dir, 'detection_results.png')
    visualize_results(processed_pcd, classified_segments, vis_output_path)
    
    # Step 6: Write results to a file
    results_file = os.path.join(standalone_output_dir, 'detection_results.txt')
    with open(results_file, 'w') as f:
        f.write("CIE Object Detection Results (Standalone Method)\n")
        f.write("==============================================\n\n")
        
        # Group by class
        class_counts = {}
        for _, class_name, confidence in classified_segments:
            if confidence >= 0.5:  # Only count medium-high confidence detections
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
        
        f.write("Object counts:\n")
        for class_name, count in class_counts.items():
            f.write(f"{class_name}: {count}\n")
        
        f.write("\nDetailed results:\n")
        for i, (segment, class_name, confidence) in enumerate(classified_segments):
            if confidence >= 0.5:  # Only include medium-high confidence detections
                bbox = segment.get_axis_aligned_bounding_box()
                center = bbox.get_center()
                min_bound = bbox.min_bound
                max_bound = bbox.max_bound
                dimensions = max_bound - min_bound
                
                f.write(f"Object {i+1}:\n")
                f.write(f"  Class: {class_name}\n")
                f.write(f"  Confidence: {confidence:.2f}\n")
                f.write(f"  Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})\n")
                f.write(f"  Dimensions: {dimensions[0]:.2f} x {dimensions[1]:.2f} x {dimensions[2]:.2f}\n")
                f.write(f"  Points: {len(segment.points)}\n\n")
    
    print(f"Results written to {results_file}")
    print(f"All standalone detection results saved to {standalone_output_dir}")
    
    print("Object detection completed successfully!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")