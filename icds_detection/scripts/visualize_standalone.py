#!/usr/bin/env python
"""
Fixed Visualization Script for Standalone Results

This script is specifically designed to parse and visualize results from the standalone detector,
with special handling for the exact format of the results file.
"""

import os
import argparse
import numpy as np
import open3d as o3d
import re

def load_pcd(pcd_path):
    """Load a point cloud file."""
    print(f"Loading point cloud from {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"Loaded {len(pcd.points)} points")
    return pcd

def parse_standalone_results(result_path):
    """
    Parse detection results from the standalone detector output.
    Specifically designed for the format used in detection_results.txt.
    """
    print(f"Parsing standalone results from {result_path}")
    
    # Parse the text file
    boxes = []
    labels = []
    scores = []
    class_names = []
    unique_classes = set()
    
    with open(result_path, 'r') as f:
        content = f.read()
    
    # Extract all object blocks using regex
    object_pattern = r"Object \d+:\s+Class: ([^\n]+)\s+Confidence: ([^\n]+)\s+Center: \(([^)]+)\)\s+Dimensions: ([^\n]+)\s+Points: (\d+)"
    matches = re.findall(object_pattern, content, re.DOTALL)
    
    print(f"Found {len(matches)} object matches in the file")
    
    for match in matches:
        try:
            # Extract class and confidence
            class_name = match[0].strip()
            confidence = float(match[1].strip())
            
            # Extract center coordinates
            center_str = match[2].strip()
            center = [float(x.strip()) for x in center_str.split(',')]
            
            # Extract dimensions
            dim_str = match[3].strip()
            dimensions = [float(x.strip()) for x in dim_str.split('x')]
            
            # Store unique class names
            if class_name not in unique_classes:
                unique_classes.add(class_name)
                class_names.append(class_name)
            
            # Add object to lists
            boxes.append(center + dimensions + [0.0])  # [x, y, z, dx, dy, dz, heading(0)]
            labels.append(class_names.index(class_name) + 1)  # 1-indexed for compatibility
            scores.append(confidence)
            
        except Exception as e:
            print(f"Error parsing object: {e} - {match}")
    
    print(f"Successfully parsed {len(boxes)} objects")
    print(f"Unique classes: {class_names}")
    
    return np.array(boxes), np.array(labels), np.array(scores), class_names

def visualize_detections(pcd, boxes, labels, scores, class_names, output_path, score_threshold=0.0):
    """
    Visualize point cloud with bounding boxes and labels.
    
    Args:
        pcd: Open3D point cloud
        boxes: Array of boxes in format [x, y, z, dx, dy, dz, heading]
        labels: Array of label indices
        scores: Array of confidence scores
        class_names: List of class names
        output_path: Path to save the visualization image
        score_threshold: Minimum score for a detection to be visualized
    """
    print(f"Visualizing {len(boxes)} detections with score threshold {score_threshold}")
    
    # Create Open3D visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Standalone Detection Results", width=1280, height=720)
    
    # Add point cloud
    vis.add_geometry(pcd)
    
    # Prepare color map for different classes
    colors = {
        'Chair': [1, 0, 0],           # Red
        'Table': [0, 1, 0],           # Green
        'Monitor': [0, 0, 1],         # Blue
        'Projection_Screen': [1, 1, 0],  # Yellow
        'Projector': [1, 0, 1],       # Magenta
        'TV': [0, 1, 1],              # Cyan
        'Computer': [0.5, 0.5, 0],    # Olive
        'Misc': [0.5, 0.5, 0.5]       # Gray
    }
    
    # Default color for unknown classes
    default_color = [0.7, 0.7, 0.7]
    
    # Count how many boxes we actually display
    displayed_count = 0
    
    # Add bounding boxes with labels
    for i, box in enumerate(boxes):
        # Skip low confidence detections
        if scores[i] < score_threshold:
            continue
        
        # Extract box parameters
        x, y, z = box[0:3]
        dx, dy, dz = box[3:6]
        heading = box[6]
        
        # Get class name
        label_idx = int(labels[i])
        if label_idx > 0 and label_idx <= len(class_names):
            class_name = class_names[label_idx-1]
        else:
            class_name = f"Class_{label_idx}"
        
        # Get color for this class
        color = colors.get(class_name, default_color)
        
        # Create oriented bounding box
        bbox = o3d.geometry.OrientedBoundingBox(
            center=[x, y, z],
            R=o3d.geometry.get_rotation_matrix_from_xyz([0, 0, heading]),
            extent=[dx, dy, dz]
        )
        bbox.color = color
        
        # Add bounding box to visualization
        vis.add_geometry(bbox)
        
        # Print box info
        print(f"Box {displayed_count+1}: {class_name} ({scores[i]:.2f}), "
              f"Center: [{x:.2f}, {y:.2f}, {z:.2f}], "
              f"Dims: [{dx:.2f}, {dy:.2f}, {dz:.2f}]")
        
        displayed_count += 1
    
    # Set rendering options
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    opt.point_size = 1.0
    
    # Set view control
    view_control = vis.get_view_control()
    # Adjust these settings based on your point cloud
    view_control.set_front([0, -1, 0])  # Look from the front
    view_control.set_lookat([0, 0, 0])  # Look at the center
    view_control.set_up([0, 0, 1])     # Z is up
    view_control.set_zoom(0.7)         # Zoom out a bit
    
    # Capture screenshot if path is provided
    if output_path:
        # Update rendering first
        vis.poll_events()
        vis.update_renderer()
        
        # Save image
        vis.capture_screen_image(output_path)
        print(f"Visualization saved to {output_path}")
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()
    
    return displayed_count

def main():
    parser = argparse.ArgumentParser(description='Visualize standalone detection results')
    parser.add_argument('--pcd_path', type=str, required=True, help='Path to point cloud file')
    parser.add_argument('--result_path', type=str, required=True, help='Path to detection results file')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save visualization image')
    parser.add_argument('--score_threshold', type=float, default=0.0, help='Minimum score threshold for visualization')
    args = parser.parse_args()
    
    # Load point cloud
    pcd = load_pcd(args.pcd_path)
    
    # Parse detection results
    boxes, labels, scores, class_names = parse_standalone_results(args.result_path)
    
    # Set default output path if not specified
    if args.output_path is None:
        args.output_path = os.path.join(os.path.dirname(args.result_path), 'visualization.png')
    
    # Visualize detections
    if len(boxes) > 0:
        box_count = visualize_detections(
            pcd, boxes, labels, scores, class_names, 
            args.output_path, 
            score_threshold=args.score_threshold
        )
        print(f"Visualization complete! Displayed {box_count} boxes above threshold {args.score_threshold}")
    else:
        print("No objects to visualize. Check the parsing results.")

if __name__ == '__main__':
    main()
