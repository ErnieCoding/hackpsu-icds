import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import os
import json
import gc

class ImprovedPointCloudProcessor:
    def __init__(self, voxel_size=0.02, room_eps=0.5, room_min_samples=100, 
                 plane_distance=0.05, min_plane_size=500, object_eps=0.05, 
                 object_min_samples=15):
        self.voxel_size = voxel_size
        self.room_eps = room_eps
        self.room_min_samples = room_min_samples
        self.plane_distance = plane_distance
        self.min_plane_size = min_plane_size
        self.object_eps = object_eps
        self.object_min_samples = object_min_samples
        self.objects = []
        self.planes = []
        
    def load_point_cloud(self, file_path):
        """Load and return a point cloud from file"""
        pcd = o3d.io.read_point_cloud(file_path)
        print(f"Loaded {file_path} with {len(pcd.points)} points")
        return pcd

    def release_memory(self):
        """Release memory by clearing stored objects"""
        self.objects = []
        self.planes = []
        gc.collect()

    def preprocess(self, pcd):
        """Preprocess the point cloud with memory efficiency in mind"""
        print(f"Original point cloud has {len(pcd.points)} points")
        
        # If point cloud is very large, use more aggressive downsampling
        voxel_size = self.voxel_size
        if len(pcd.points) > 10000000:  # 10 million points
            voxel_size = self.voxel_size * 2
            print(f"Large point cloud detected. Using more aggressive voxel size: {voxel_size}")
        
        # Downsample using voxel grid
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"Downsampled to {len(pcd_down.points)} points")
        
        # Skip normal estimation to save memory
        print("Skipping normal estimation to conserve memory")
        
        # For outlier removal, use minimal parameters
        nb_neighbors = 10
        std_ratio = 3.0
        print("Using faster outlier removal parameters")
        
        # Fix function name: remove_statistical_outlier (not outliers)
        pcd_clean, _ = pcd_down.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        print(f"After outlier removal: {len(pcd_clean.points)} points")
        
        return pcd_clean
    
    def segment_room(self, pcd):
        """Segment the point cloud into room-level segments with extremely memory-efficient approach"""
        points = np.asarray(pcd.points)
        total_points = len(points)
        
        # If point cloud is too large, process in smaller chunks
        if total_points > 100000:  # Lower threshold
            print(f"Large point cloud detected ({total_points} points). Processing in smaller chunks...")
            
            # Spatial chunking - divide space into grid cells
            min_bound = pcd.get_min_bound()
            max_bound = pcd.get_max_bound()
            extent = max_bound - min_bound
            
            # Use a larger grid for more manageable chunks
            grid_size = max(4, int(np.cbrt(total_points / 50000)))  # Aim for 50K points per chunk
            print(f"Dividing space into {grid_size}x{grid_size}x{grid_size} grid")
            
            # Create chunks and process each
            room_segments = []
            chunk_id = 0
            
            # Process one grid cell at a time to minimize memory usage
            for x in range(grid_size):
                x_min = min_bound[0] + x * extent[0] / grid_size
                x_max = min_bound[0] + (x + 1) * extent[0] / grid_size
                
                for y in range(grid_size):
                    y_min = min_bound[1] + y * extent[1] / grid_size
                    y_max = min_bound[1] + (y + 1) * extent[1] / grid_size
                    
                    for z in range(1):  # Process only one z-level at a time to reduce memory
                        z_min = min_bound[2] + z * extent[2] / grid_size
                        z_max = min_bound[2] + (z + 1) * extent[2] / grid_size
                        
                        # Create chunk bounding box
                        chunk_min = np.array([x_min, y_min, z_min])
                        chunk_max = np.array([x_max, y_max, z_max])
                        
                        # Select points in this chunk
                        chunk_indices = np.where(
                            (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
                            (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
                            (points[:, 2] >= z_min) & (points[:, 2] < z_max)
                        )[0]
                        
                        if len(chunk_indices) < self.room_min_samples:
                            continue
                        
                        chunk_pcd = pcd.select_by_index(chunk_indices)
                        print(f"Processing chunk {chunk_id} with {len(chunk_indices)} points...")
                        
                        # If chunk is still too large, downsample further
                        if len(chunk_indices) > 100000:
                            chunk_pcd = chunk_pcd.voxel_down_sample(voxel_size=self.voxel_size * 2)
                            print(f"Downsampled large chunk to {len(chunk_pcd.points)} points")
                        
                        # Process this chunk with DBSCAN
                        try:
                            chunk_points = np.asarray(chunk_pcd.points)
                            db = DBSCAN(eps=self.room_eps, min_samples=self.room_min_samples, n_jobs=-1).fit(chunk_points)
                            chunk_labels = db.labels_
                            
                            # Extract room segments from this chunk
                            unique_labels = set(chunk_labels)
                            for label in unique_labels:
                                if label == -1:  # Skip noise
                                    continue
                                
                                # Get points in this segment
                                segment_indices = np.where(chunk_labels == label)[0]
                                segment_pcd = chunk_pcd.select_by_index(segment_indices)
                                
                                room_segments.append({
                                    'pcd': segment_pcd,
                                    'label': chunk_id * 1000 + label,  # Ensure unique labels across chunks
                                    'size': len(segment_indices)
                                })
                            
                        except MemoryError:
                            print(f"Memory error processing chunk {chunk_id}. Skipping...")
                            continue
                        
                        # Force garbage collection between chunks
                        db = None
                        chunk_points = None
                        chunk_labels = None
                        gc.collect()
                        
                        chunk_id += 1
            
            print(f"Found {len(room_segments)} room segments across all chunks")
            return room_segments
    
        else:
            # For smaller point clouds, use the original approach
            db = DBSCAN(eps=self.room_eps, min_samples=self.room_min_samples, n_jobs=-1).fit(points)
            labels = db.labels_
            
            # Get unique labels
            unique_labels = set(labels)
            n_rooms = len(unique_labels) - (1 if -1 in labels else 0)
            
            print(f"Found {n_rooms} room segments")
            
            # Create room segments
            room_segments = []
            for label in unique_labels:
                if label == -1:  # Skip noise
                    continue
                    
                # Get points in room segment
                room_indices = np.where(labels == label)[0]
                room_pcd = pcd.select_by_index(room_indices)
                
                room_segments.append({
                    'pcd': room_pcd,
                    'label': label,
                    'size': len(room_indices)
                })
                
            return room_segments
    
    def extract_major_planes(self, pcd, max_planes=6):
        """Extract major planes (walls, floors, ceilings) from point cloud"""
        planes = []
        # remaining_pcd = pcd.copy()
        # Create a new point cloud with the same data
        remaining_pcd = o3d.geometry.PointCloud()
        remaining_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
        if pcd.has_normals():
            remaining_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals))
        if pcd.has_colors():
            remaining_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
        
        # Extract planes until no more significant planes can be found
        for i in range(max_planes):
            if len(remaining_pcd.points) < self.min_plane_size:
                break
                
            # Segment plane
            plane_model, inliers = remaining_pcd.segment_plane(
                distance_threshold=self.plane_distance,
                ransac_n=3,
                num_iterations=1000
            )
            
            if len(inliers) < self.min_plane_size:
                break
                
            # Extract plane and add to planes list
            plane_pcd = remaining_pcd.select_by_index(inliers)
            planes.append({
                'pcd': plane_pcd,
                'model': plane_model,
                'size': len(inliers)
            })
            
            # Remove plane points from remaining points
            remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
            
            print(f"Extracted plane #{i+1} with {len(inliers)} points. {len(remaining_pcd.points)} points remaining.")
            
        return planes, remaining_pcd   
    
    def detect_objects_in_room(self, room_pcd):
        """Detect objects in a room segment after removing structural elements"""
        # First, extract major planes (walls, floors, etc.)
        planes, remaining_pcd = self.extract_major_planes(room_pcd)
        
        # If the point cloud is very large, downsample further
        if len(remaining_pcd.points) > 500000:
            print(f"Further downsampling large point cloud from {len(remaining_pcd.points)} points...")
            remaining_pcd = remaining_pcd.voxel_down_sample(voxel_size=self.voxel_size * 2)
            print(f"Downsampled to {len(remaining_pcd.points)} points")
        
        print(f"Remaining points after plane removal: {len(remaining_pcd.points)}")
        
        # Now cluster the remaining points to find objects
        points = np.asarray(remaining_pcd.points)
        
        # Use batch processing for DBSCAN if point cloud is large
        if len(points) > 100000:
            print("Using batch processing for DBSCAN...")
            
            # Divide space into quadrants (spatial chunking)
            min_bound = remaining_pcd.get_min_bound()
            max_bound = remaining_pcd.get_max_bound()
            mid_point = (min_bound + max_bound) / 2
            
            # Create quadrants
            quadrants = [
                np.where((points[:, 0] < mid_point[0]) & (points[:, 1] < mid_point[1]))[0],
                np.where((points[:, 0] >= mid_point[0]) & (points[:, 1] < mid_point[1]))[0],
                np.where((points[:, 0] < mid_point[0]) & (points[:, 1] >= mid_point[1]))[0],
                np.where((points[:, 0] >= mid_point[0]) & (points[:, 1] >= mid_point[1]))[0]
            ]
            
            # Process each quadrant
            all_labels = np.full(len(points), -1, dtype=int)
            current_max_label = -1
            
            for q_idx, quadrant_indices in enumerate(quadrants):
                if len(quadrant_indices) < self.object_min_samples:
                    continue
                    
                print(f"Processing quadrant {q_idx+1} with {len(quadrant_indices)} points...")
                
                # Run DBSCAN on this quadrant
                quadrant_points = points[quadrant_indices]
                db = DBSCAN(eps=self.object_eps, min_samples=self.object_min_samples, n_jobs=-1).fit(quadrant_points)
                quadrant_labels = db.labels_
                
                # Update labels, ensuring they don't overlap across quadrants
                valid_indices = np.where(quadrant_labels != -1)[0]
                all_labels[quadrant_indices[valid_indices]] = quadrant_labels[valid_indices] + current_max_label + 1
                
                if len(valid_indices) > 0:
                    current_max_label = np.max(all_labels)
            
            labels = all_labels
        else:
            # For smaller point clouds, use standard DBSCAN
            db = DBSCAN(eps=self.object_eps, min_samples=self.object_min_samples, n_jobs=-1).fit(points)
            labels = db.labels_
        
        # Get unique labels
        unique_labels = set(labels)
        n_objects = len(unique_labels) - (1 if -1 in labels else 0)
        
        print(f"Found {n_objects} potential objects")
        
        # Create object segments
        room_objects = []
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
                
            # Get points in object segment
            object_indices = np.where(labels == label)[0]
            object_pcd = remaining_pcd.select_by_index(object_indices)
            
            # Skip if too few points
            if len(object_indices) < self.object_min_samples:
                continue
            
            # Compute bounding box
            bbox = object_pcd.get_axis_aligned_bounding_box()
            bbox_points = np.asarray(bbox.get_box_points())
            
            # Get dimensions
            dimensions = np.ptp(bbox_points, axis=0)  # max - min in each dimension
            
            # Add object with properties
            room_objects.append({
                'pcd': object_pcd,
                'bbox': bbox,
                'dimensions': dimensions,
                'num_points': len(object_indices),
                'label': label,
                'classification': None
            })
        
        return room_objects, planes
    
    def classify_objects(self, objects, room_height=11.0):
        """Classify objects based on dimensions and position"""
        # Print debug info
        print(f"Starting classification of {len(objects)} objects")
        
        # Define dimension ranges for each object class (in feet)
        # Using the provided measurements, with some tolerance
        class_dimensions = {
            'chair': {
                'x': (0.8, 2.5),    # Increase tolerance range
                'y': (0.8, 2.5),    
                'z': (2.0, 5.0)     
            },
            'table': {
                'x': (1.5, 6.0),    # Increase tolerance range
                'y': (1.5, 6.0),    
                'z': (2.0, 5.0)     
            },
            'monitor': {
                'x': (0.8, 2.5),    # Increase tolerance range
                'y': (0.1, 1.0),    
                'z': (1.0, 2.5)     
            },
            'screen': {
                'x': (5.0, 10.0),   # Increase tolerance range
                'y': (0.1, 1.0),    
                'z': (3.0, 6.0)     
            },
            'projector': {
                'x': (0.8, 2.0),    # Increase tolerance range
                'y': (0.5, 1.5),    
                'z': (0.2, 1.0)     
            }
        }
        
        # Function to check if dimensions match a class
        def dimensions_match(dims, class_spec):
            # Sort dimensions to handle orientation-agnostic matching
            sorted_dims = np.sort(dims)
            class_dims = np.array([
                [class_spec['x'][0], class_spec['y'][0], class_spec['z'][0]],
                [class_spec['x'][1], class_spec['y'][1], class_spec['z'][1]]
            ])
            sorted_class_mins = np.sort(class_dims[0])
            sorted_class_maxs = np.sort(class_dims[1])
            
            # Check if sorted dimensions fall within the sorted class ranges
            return np.all(sorted_dims >= sorted_class_mins) and np.all(sorted_dims <= sorted_class_maxs)
        
        # Classify each object
        classified_objects = []
        for obj in objects:
            dimensions = obj['dimensions']
            center = obj['bbox'].get_center()
            
            # For debugging
            print(f"Object dimensions: {dimensions}")
            
            # MODIFICATION: Adjust size filtering criteria
            # Skip objects that are too large (likely structural)
            if np.any(dimensions > room_height * 0.9):  # Changed from 0.8 to 0.9
                print(f"Skipping object with dimensions {dimensions} - too large")
                continue
                
            # Skip objects that are too small (noise)
            if np.all(dimensions < 0.3):  # Changed from 0.5 to 0.3
                print(f"Skipping object with dimensions {dimensions} - too small")
                continue
                
            # Initialize with unknown classification - but include all objects
            classification = 'unknown'
            confidence = 0.0
            
            # Try to match with known objects
            for class_name, class_spec in class_dimensions.items():
                if dimensions_match(dimensions, class_spec):
                    classification = class_name
                    
                    # Calculate a simple confidence based on how well dimensions match
                    sorted_dims = np.sort(dimensions)
                    class_dims = np.array([
                        [class_spec['x'][0], class_spec['y'][0], class_spec['z'][0]],
                        [class_spec['x'][1], class_spec['y'][1], class_spec['z'][1]]
                    ])
                    sorted_class_mins = np.sort(class_dims[0])
                    sorted_class_maxs = np.sort(class_dims[1])
                    
                    # Calculate how centered the dimensions are in their respective ranges
                    range_positions = (sorted_dims - sorted_class_mins) / (sorted_class_maxs - sorted_class_mins)
                    # Confidence is highest when dimensions are in the middle of the range (0.5)
                    confidence = 1.0 - 2.0 * np.abs(range_positions - 0.5).mean()
                    
                    break
            
            # Add classification to object
            obj['classification'] = classification
            obj['confidence'] = confidence
            classified_objects.append(obj)
            print(f"Classified as {classification} with confidence {confidence:.2f}")
            
        print(f"Classified {len(classified_objects)} objects")
        return classified_objects
    
    def process_full_pipeline(self, file_path):
        """Run the full pipeline on a point cloud file"""
        # Load point cloud
        pcd = self.load_point_cloud(file_path)
        
        # Preprocess
        print("Preprocessing point cloud...")
        pcd_clean = self.preprocess(pcd)
        
        # Segment into rooms
        print("Segmenting into rooms...")
        room_segments = self.segment_room(pcd_clean)
        
        # Process each room
        all_objects = []
        all_planes = []
        
        for i, room in enumerate(room_segments):
            print(f"Processing room {i+1}/{len(room_segments)}")
            
            # Detect objects in room
            room_objects, planes = self.detect_objects_in_room(room['pcd'])
            
            # Classify objects
            print(f"Classifying objects in room {i+1}...")
            classified_objects = self.classify_objects(room_objects)
            
            # Add room index to objects
            for obj in classified_objects:
                obj['room_idx'] = i
                
            # Add to all objects
            all_objects.extend(classified_objects)
            all_planes.extend(planes)
            
        self.objects = all_objects
        self.planes = all_planes
        return all_objects
    
    def save_detection_results(self, output_path):
        """Save detection results to file in the format expected by the frontend"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"Saving {len(self.objects)} objects to {output_path}")
        
        # Format results in the expected format
        results = []
        for i, obj in enumerate(self.objects):
            # Extract bbox corners
            bbox = obj['bbox']
            corners = np.asarray(bbox.get_box_points())
            min_bound = bbox.get_min_bound()
            max_bound = bbox.get_max_bound()
            center = bbox.get_center()
            
            # Create result entry in expected format
            result = {
                'id': i,
                'label': obj['classification'],
                'confidence': float(obj['confidence']),
                'dimensions': {
                    'width': float(obj['dimensions'][0]),
                    'depth': float(obj['dimensions'][1]),
                    'height': float(obj['dimensions'][2])
                },
                'position': {
                    'x': float(center[0]),
                    'y': float(center[1]),
                    'z': float(center[2])
                },
                'bbox': {
                    'min_x': float(min_bound[0]),
                    'min_y': float(min_bound[1]),
                    'min_z': float(min_bound[2]),
                    'max_x': float(max_bound[0]),
                    'max_y': float(max_bound[1]),
                    'max_z': float(max_bound[2])
                }
            }
            results.append(result)
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Detection results saved to {output_path}")
