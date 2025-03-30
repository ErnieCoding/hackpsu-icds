# CIE Point Cloud Object Detection

## Project Overview

This project implements an automated 3D point cloud object detection system for the Center for Immersive Experiences (CIE) at Penn State. The system analyzes 3D point cloud data of the CIE space to identify and classify objects such as chairs, tables, monitors, projection screens, and other furniture items.

## Features

- **Point Cloud Processing**: Loads and processes 3D point cloud data (.pcd files)
- **Hierarchical DBSCAN Segmentation**: Implements a multi-level approach to identify objects
- **Plane Extraction**: Removes walls, floors, and ceilings before object detection
- **Object Classification**: Classifies detected objects based on dimensions and color
- **Visualization**: Interactive 3D visualization of point clouds and detected objects

## Frontend Interface

The web-based frontend provides an intuitive interface for uploading, analyzing, and visualizing point cloud data:

- **Upload Interface**: Allows users to upload .pcd files for analysis
- **Data Summary**: Shows point cloud statistics including total points and sample data
- **Object Identification**: Lists all detected objects with confidence scores
- **Object Details**: Displays detailed information about selected objects, including:
  - Dimensions (width, depth, height)
  - Position coordinates (X, Y, Z)
- **3D Visualization**: Interactive viewer for the point cloud with highlighted detected objects
- **Plot Options**: Customization options for the visualization

## Technical Implementation

The system consists of several core components:

1. **Preprocessing**: Downsampling, outlier removal, and normal estimation
2. **Room Segmentation**: Large-scale DBSCAN clustering to identify separate spaces
3. **Plane Extraction**: RANSAC-based plane fitting to remove structural elements
4. **Object Detection**: Fine-grained DBSCAN clustering to identify potential objects
5. **Classification**: Dimension and color-based classification system
6. **Visualization**: Interactive 3D visualization with Open3D

## Challenges and Limitations

The implementation faced several significant challenges:

1. **Memory Management**: Processing large point clouds required careful memory optimization through:
   - Chunked processing
   - Progressive garbage collection
   - Adaptive downsampling

2. **Classification Accuracy**: The current classification system struggles with:
   - Distinguishing objects with similar dimensions
   - Identifying objects in unusual orientations
   - Handling objects that don't match predefined dimensional criteria

3. **Parameter Sensitivity**: The DBSCAN algorithm is highly sensitive to parameter choices:
   - Different room areas require different eps and min_points values
   - A single parameter set doesn't work well across the entire CIE space

4. **Structural Element Confusion**: Despite plane extraction, some structural elements are still mistakenly classified as objects.

## Future Improvements

Given more time, several enhancements could significantly improve the system:

1. **Machine Learning Classification**: Replace rule-based classification with a trained model:
   - Extract feature vectors from detected objects
   - Train a classifier (Random Forest, SVM, etc.) on labeled examples
   - Implement feature-based classification instead of rigid dimensional rules

2. **Adaptive Parameter Selection**: Implement adaptive parameter selection based on point density:
   - Analyze local point density before running DBSCAN
   - Adjust eps and min_points automatically for different regions

3. **Contextual Analysis**: Incorporate spatial relationships between objects:
   - Identify common arrangements (e.g., monitors on desks)
   - Use relative positions to improve classification
   - Group related objects together

4. **Color-Based Refinement**: Better leverage color information:
   - Create color histograms for object types
   - Use color distribution patterns for classification
   - Implement color-based segmentation for ambiguous objects

5. **Deep Learning Integration**: For state-of-the-art performance:
   - Integrate with OpenPCDet or similar frameworks
   - Train on CIE-specific data
   - Implement end-to-end 3D object detection

## Usage

### Detection

```bash
python scripts/cie_detection.py --input_dir data/cie --output_dir output --eps 0.1 --min_points 50 --voxel_size 0.04
```

### Visualization

```bash
python scripts/visualize_standalone.py --pcd_path data/cie_processed/CIE.pcd --result_path output/standalone_detection/detection_results.txt
```

## Requirements

- Python 3.6+
- Open3D
- NumPy
- scikit-learn
- Matplotlib

## Acknowledgments

This project was developed as part of the ICDS Challenge at HackPSU for the Center for Immersive Experiences (CIE) at Penn State.

## Conclusion

While the current implementation provides a functional foundation for point cloud object detection in the CIE space, the results fall short of the desired accuracy. With the proposed improvements, particularly the integration of machine learning techniques and contextual analysis, the system could achieve much higher detection and classification accuracy.
