#!/usr/bin/env python
"""
Improved CIE Point Cloud Object Detection

This script:
1. Loads and preprocesses the point cloud data
2. Segments the point cloud to identify potential objects
3. Classifies each segment based on geometric features
4. Visualizes and saves the results
"""

import os
import argparse
import glob
from improved_segmentation import ImprovedPointCloudProcessor
import time
import gc  # Garbage collection
import open3d as o3d
import numpy as np

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Point Cloud Object Detection')
    parser.add_argument('--input_dir', required=True, help='Input directory containing point cloud files')
    parser.add_argument('--output_dir', required=True, help='Output directory for detection results')
    parser.add_argument('--eps', type=float, default=0.1, help='DBSCAN eps parameter for object detection')
    parser.add_argument('--min_points', type=int, default=50, help='DBSCAN min_points parameter for object detection')
    parser.add_argument('--voxel_size', type=float, default=0.04, help='Voxel size for downsampling')
    args = parser.parse_args()
    
    # Find all point cloud files
    point_cloud_files = glob.glob(os.path.join(args.input_dir, '*.pcd'))
    if not point_cloud_files:
        print(f"No point cloud files found in {args.input_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'standalone_detection'), exist_ok=True)
    
    # For memory efficiency, process only part of the point cloud for large files
    for pcd_file in point_cloud_files:
        print(f"Processing {pcd_file}...")
        
        # Get file size
        file_size = os.path.getsize(pcd_file) / (1024 * 1024)  # Size in MB
        print(f"File size: {file_size:.2f} MB")
        
        # For very large files, sample the point cloud
        pcd = o3d.io.read_point_cloud(pcd_file)
        original_points = len(pcd.points)
        
        # If extremely large, sample points
        if original_points > 5000000:  # 5 million points
            print(f"Very large point cloud ({original_points} points). Sampling 20%...")
            pcd = pcd.random_down_sample(0.2)
            print(f"Sampled to {len(pcd.points)} points")
        
        # Create processor with parameters from command line
        processor = ImprovedPointCloudProcessor(
            voxel_size=args.voxel_size,
            room_eps=args.eps * 10,
            room_min_samples=args.min_points * 2,
            object_eps=args.eps,
            object_min_samples=args.min_points
        )
        
        # Process in stages to manage memory
        start_time = time.time()
        
        # Preprocess
        print("Preprocessing point cloud...")
        pcd_clean = processor.preprocess(pcd)
        
        # Free memory
        del pcd
        gc.collect()
        
        # Segment into rooms
        print("Segmenting into rooms...")
        room_segments = processor.segment_room(pcd_clean)
        
        # Free memory
        del pcd_clean
        gc.collect()
        
        # Process each room and accumulate results
        all_objects = []
        
        # Process only a few rooms at a time to save memory
        room_batch_size = 5
        for i in range(0, len(room_segments), room_batch_size):
            batch = room_segments[i:i + room_batch_size]
            print(f"Processing room batch {i//room_batch_size + 1}/{(len(room_segments) + room_batch_size - 1)//room_batch_size}")
            
            for j, room in enumerate(batch):
                room_index = i + j
                print(f"Processing room {room_index+1}/{len(room_segments)}")
                
                # If room is too large, downsample further
                if len(room['pcd'].points) > 100000:
                    print(f"Large room with {len(room['pcd'].points)} points. Downsampling...")
                    room['pcd'] = room['pcd'].voxel_down_sample(voxel_size=args.voxel_size * 2)
                    print(f"Downsampled to {len(room['pcd'].points)} points")
                
                # Detect objects in room
                try:
                    room_objects, planes = processor.detect_objects_in_room(room['pcd'])
                    
                    # Free memory
                    del planes
                    gc.collect()
                    
                    # Classify objects
                    print(f"Classifying objects in room {room_index+1}...")
                    classified_objects = processor.classify_objects(room_objects)
                    
                    # Add room index to objects
                    for obj in classified_objects:
                        obj['room_idx'] = room_index
                        
                    # Add to all objects
                    all_objects.extend(classified_objects)
                    
                    # Free memory
                    del room_objects
                    del classified_objects
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error processing room {room_index+1}: {e}")
                    continue
            
            # Force garbage collection between batches
            gc.collect()
        
        # Save detection results
        processor.objects = all_objects
        output_path = os.path.join(args.output_dir, 'standalone_detection', 'detection_results.txt')
        processor.save_detection_results(output_path)
        
        elapsed_time = time.time() - start_time
        
        # Print statistics
        object_classes = [obj['classification'] for obj in all_objects]
        class_counts = {cls: object_classes.count(cls) for cls in set(object_classes)}
        
        print(f"Processed {pcd_file} in {elapsed_time:.2f} seconds")
        print("Detection results:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count}")
        print(f"Total objects: {len(all_objects)}")
        print(f"Results saved to {output_path}")
        
        # Free memory before processing next file
        del all_objects
        del processor
        gc.collect()


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
