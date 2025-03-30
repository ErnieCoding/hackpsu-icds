#!/usr/bin/env python
"""
Visualize Detection Results with Labels

This script:
1. Loads a point cloud file
2. Loads detection results from either standalone or OpenPCDet output
3. Visualizes the point cloud with labeled 3D bounding boxes
4. Saves the visualization to an image file
"""

import os
import sys
import argparse
import numpy as np
import pickle
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

def load_pcd(pcd_path):
    """Load a point cloud file."""
    print(f"Loading point cloud from {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"Loaded {len(pcd.points)} points")
    return pcd

def load_openpcdet_results(result_path):
    """Load detection results from OpenPCDet output."""
    print(f"Loading OpenPCDet results from {result_path}")
    with open(result_path, 'rb') as f:
        results = pickle.load(f)
    
    # Extract boxes, labels, and scores
    boxes = results.get('boxes_lidar', [])  # [x, y, z, dx, dy, dz, heading]
    labels = results.get('pred_labels', results.get('labels', []))
    scores = results.get('scores', [])
    names = results.get('name', results.get('class_names', []))
    
    if not isinstance(names, list) and len(boxes) > 0:
        # Convert single class name to list if needed
        names = [names[i-1] if i-1 < len(names) else f"Class_{i}" for i in labels]
    
    print(f"Loaded {len(boxes)} detections")
    return boxes, labels, scores, names

def load_standalone_results(result_path):
    """Load detection results from standalone detector output."""
    print(f"Loading standalone results from {result_path}")
       # Check if file exists
    if not os.path.exists(result_path):
        print(f"ERROR: Result file not found: {result_path}")
        return np.array([]), np.array([]), np.array([]), []
    # Parse the text file
    boxes = []
    labels = []
    scores = []
    names = []
    
    with open(result_path, 'r') as f:
        lines = f.readlines()
        
        in_object_section = False
        current_object = {}
        
        for line in lines:
            line = line.strip()
            print(f"Read {len(lines)} lines from file")  
            # Print first few lines for debugging
            print("First 10 lines:")
            for i in range(min(10, len(lines))):
                print(f"  {lines[i].strip()}")
            in_object_section = False
            current_object = {}
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith('Object '):
                    in_object_section = True
                    current_object = {}
                    print(f"Found object at line {i}: {line}")
                elif in_object_section and line.startswith('  Class:'):
                    current_object['class'] = line.split(':')[1].strip()
                    print(f"  Found class: {current_object['class']}")
                elif in_object_section and line.startswith('  Confidence:'):
                    current_object['confidence'] = float(line.split(':')[1].strip())
                elif in_object_section and line.startswith('  Center:'):
                    try:
                        center_str = line.split('(')[1].split(')')[0]
                        center = [float(x) for x in center_str.split(',')]
                        current_object['center'] = center
                        print(f"  Found center: {center}")
                    except Exception as e:
                        print(f"  Error parsing center at line {i}: {line}")
                        print(f"  Error: {e}")
                elif in_object_section and line.startswith('  Dimensions:'):
                    try:
                        dim_parts = line.split(':')[1].strip().split('x')
                        dimensions = [float(x.strip()) for x in dim_parts]
                        current_object['dimensions'] = dimensions
                        print(f"  Found dimensions: {dimensions}")
                    except Exception as e:
                        print(f"  Error parsing dimensions at line {i}: {line}")
                        print(f"  Error: {e}")
                elif in_object_section and line.startswith('  Points:'):
                    # End of current object
                    if all(k in current_object for k in ['class', 'center', 'dimensions']):
                        # Create a box in format [x, y, z, dx, dy, dz, heading(0)]
                        box = current_object['center'] + current_object['dimensions'] + [0.0]
                        boxes.append(box)
                        labels.append(len(names) + 1)
                        scores.append(current_object.get('confidence', 1.0))
                        names.append(current_object['class'])
                        print(f"  Added object: {current_object['class']}")
                    else:
                        print(f"  Missing required fields: {current_object}")
                    
                    in_object_section = False
    print(f"Loaded {len(boxes)} detections")
    print(f"Names: {names[:5]}...")
    print(f"First few boxes: {boxes[:2]}")
    return np.array(boxes), np.array(labels), np.array(scores), names

def visualize_detections(pcd, boxes, labels, scores, names, output_path, score_threshold=0.3, show_scores=True):
    """
    Visualize point cloud with bounding boxes and labels.
    
    Args:
        pcd: Open3D point cloud
        boxes: Array of boxes in format [x, y, z, dx, dy, dz, heading]
        labels: Array of label indices
        scores: Array of confidence scores
        names: List or array of class names
        output_path: Path to save the visualization image
        score_threshold: Minimum score for a detection to be visualized
        show_scores: Whether to show confidence scores on labels
    """
    print(f"Visualizing detections with score threshold {score_threshold}")
    
    # Create Open3D visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)
    
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
    
    # Add bounding boxes with labels
    box_count = 0
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
        if label_idx > 0 and label_idx <= len(names):
            class_name = names[label_idx-1]
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
        
        # Prepare label text
        if show_scores:
            label_text = f"{class_name}: {scores[i]:.2f}"
        else:
            label_text = class_name
        
        print(f"Box {box_count+1}: {label_text}, Center: [{x:.2f}, {y:.2f}, {z:.2f}], "
              f"Dims: [{dx:.2f}, {dy:.2f}, {dz:.2f}], Heading: {heading:.2f}")
        
        box_count += 1
    
    # Set rendering options
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    render_option.point_size = 1.0
    
    # Update camera view
    view_control = vis.get_view_control()
    view_control.set_front([0, -1, 0])  # Look from the front
    view_control.set_lookat([0, 0, 0])  # Look at the center
    view_control.set_up([0, 0, 1])     # Z is up
    view_control.set_zoom(0.7)         # Zoom out a bit
    
    # Update visualization
    vis.poll_events()
    vis.update_renderer()
    
    # Capture screenshot
    if output_path:
        vis.capture_screen_image(output_path)
        print(f"Visualization saved to {output_path}")
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()
    
    return box_count

def main():
    parser = argparse.ArgumentParser(description='Visualize detection results with labels')
    parser.add_argument('--pcd_path', type=str, required=True, help='Path to point cloud file')
    parser.add_argument('--result_path', type=str, required=True, help='Path to detection results file')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save visualization image')
    parser.add_argument('--type', type=str, choices=['openpcdet', 'standalone'], default='openpcdet',
                       help='Type of detection results to load')
    parser.add_argument('--score_threshold', type=float, default=0.3, help='Minimum score threshold for visualization')
    parser.add_argument('--no_scores', action='store_true', help='Hide confidence scores on labels')
    args = parser.parse_args()
    
    # Load point cloud
    pcd = load_pcd(args.pcd_path)
    
    # Load detection results
    if args.type == 'openpcdet':
        boxes, labels, scores, names = load_openpcdet_results(args.result_path)
    else:
        boxes, labels, scores, names = load_standalone_results(args.result_path)
    
    # Determine output path if not specified
    if args.output_path is None:
        dirname = os.path.dirname(args.result_path)
        basename = os.path.basename(args.result_path).split('.')[0]
        args.output_path = os.path.join(dirname, f"{basename}_labeled_visualization.png")
    
    # Visualize detections
    box_count = visualize_detections(
        pcd, boxes, labels, scores, names, 
        args.output_path, 
        score_threshold=args.score_threshold,
        show_scores=not args.no_scores
    )
    
    print(f"Visualization complete! Displayed {box_count} boxes above threshold {args.score_threshold}")
    # After loading results, add these lines:
    print("Boxes:", boxes)
    print("Labels:", labels)
    print("Scores:", scores)
    print("Names:", names)

if __name__ == '__main__':
    main()
