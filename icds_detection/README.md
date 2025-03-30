CDS Challenge: Point Cloud Object Detection

This project implements an object detection system for the Center for Immersive Experiences (CIE) point cloud data, as part of the ICDS Challenge.

## Overview

We've developed a solution to detect objects such as chairs, tables, monitors, projection screens, and other equipment in the CIE space using 3D point cloud data. Our approach leverages the OpenPCDet framework, a powerful open-source library for LiDAR-based 3D object detection.

![Point Cloud Visualization](https://immersive.psu.edu/spaces/)

## Color codes
- Chair: Red
- Table: Green
- Monitor: Blue
- Projection Screen: Yellow
- Projector: Magenta
- TV: Cyan
- Computer: Olive
- Misc/Other: Gray

## Key Features

- **Data Processing**: Handles the merging and preprocessing of multiple point cloud files for the CIE space
- **Object Detection**: Identifies and classifies objects in the 3D space using state-of-the-art detection models
- **Visualization**: Provides 3D visualization of the detected objects with bounding boxes and labels

## Technical Approach

### 1. Data Processing

The CIE point cloud data is provided in multiple PCD files for different areas of the space. Our solution:

- Merges these files into a single cohesive point cloud using the exact same approach as the provided Colab script
- Processes the four main sections (ceilingtheater, theater, main, secondary) as specified in the challenge
- Preprocesses the data to ensure compatibility with detection models

### 2. Object Detection Framework

We use OpenPCDet as our primary framework, which provides:

- A clear and unified data-model separation
- Support for multiple state-of-the-art 3D detection models
- Efficient and GPU-accelerated processing

### 3. Model Selection

For this challenge, we implemented and compared multiple detection models:

- **PointPillar**: Fast and efficient detection with good accuracy for indoor scenes
- **SECOND**: Provides improved detection accuracy with sparse convolution
- **PV-RCNN**: Offers high-precision detection with point-voxel feature fusion

### 4. Custom Dataset Integration

We created a custom CIE dataset class that:
- Inherits from OpenPCDet's DatasetTemplate
- Provides proper point cloud feature encoding
- Handles the specific characteristics of the CIE space

### 5. Visualization

Our solution includes visualization tools that:
- Display the point cloud with detected objects
- Render 3D bounding boxes with class labels
- Allow for interactive exploration of the results

## Results

Our implementation successfully detects various objects in the CIE space, including:

- Chairs and tables
- Monitors and projection screens
- Projectors and TVs
- Computers and miscellaneous objects

The detection results are visualized with color-coded 3D bounding boxes, where each color represents a different object class.

## Implementation Details

### Directory Structure

```
├── configs/
│   ├── cie_dataset.yaml         # CIE dataset configuration
│   └── cie_pointpillar.yaml     # PointPillar model configuration for CIE
├── data/
│   └── cie/                     # CIE point cloud data
│       ├── ceilingtheater/      # Theater ceiling PCD files
│       ├── theater/             # Theater area PCD files
│       ├── main/                # Main section PCD files
│       └── secondary/           # Secondary area PCD files
├── output/                      # Detection results
├── pcdet/                       # OpenPCDet library (if using)
└── scripts/
    ├── cie_detection.py         # Main detection script
    ├── merge_colab_script.py    # PCD merging script (based on the Colab notebook)
    └── prepare_dataset.py       # Dataset preparation for OpenPCDet
```

### Usage

1. **Install Dependencies**:
   ```bash
   pip install open3d numpy torch scikit-learn tqdm
   # Optional for OpenPCDet
   pip install spconv-cu118  # Match with your CUDA version
   ```

2. **Merge PCD Files** using the Colab script logic:
   ```bash
   python scripts/merge_colab_script.py --input_dir data/cie --output_path data/cie/CIE.pcd --visualize
   ```

3. **Run Standalone Object Detection**:
   ```bash
   python scripts/cie_detection.py --input_dir data/cie --output_dir output
   ```

4. **For OpenPCDet Integration** (optional):
   ```bash
   # First prepare the dataset
   python scripts/prepare_dataset.py --input_dir data/cie --output_dir data/cie_processed

   # Then run detection with OpenPCDet model
   python scripts/cie_detection.py --input_dir data/cie --output_dir output --cfg_file configs/cie_pointpillar.yaml --ckpt checkpoints/latest.pth
   ```

## Future Work

Our solution could be further improved with:

1. Fine-tuning models specifically for the CIE environment
2. Implementing real-time detection for interactive applications
3. Adding semantic segmentation for more detailed scene understanding
4. Integrating with VR/AR applications for enhanced immersive experiences

## Team

This project was developed for the ICDS Challenge at HackPSU by our team.

## Acknowledgements

- Penn State Center for Immersive Experiences for providing the point cloud data
- OpenPCDet development team for their excellent 3D detection framework
- Open3D library for point cloud processing and visualization# ICDS Challenge: Point Cloud Object Detection

This project implements an object detection system for the Center for Immersive Experiences (CIE) point cloud data, as part of the ICDS Challenge.

## Overview

We've developed a solution to detect objects such as chairs, tables, monitors, projection screens, and other equipment in the CIE space using 3D point cloud data. Our approach leverages the OpenPCDet framework, a powerful open-source library for LiDAR-based 3D object detection.

![Point Cloud Visualization](https://immersive.psu.edu/spaces/)

## Key Features

- **Data Processing**: Handles the merging and preprocessing of multiple point cloud files for the CIE space
- **Object Detection**: Identifies and classifies objects in the 3D space using state-of-the-art detection models
- **Visualization**: Provides 3D visualization of the detected objects with bounding boxes and labels

## Technical Approach

### 1. Data Processing

The CIE point cloud data is provided in multiple PCD files for different areas of the space. Our solution:

- Merges these files into a single cohesive point cloud using the exact same approach as the provided Colab script
- Processes the four main sections (ceilingtheater, theater, main, secondary) as specified in the challenge
- Preprocesses the data to ensure compatibility with detection models

### 2. Object Detection Framework

We use OpenPCDet as our primary framework, which provides:

- A clear and unified data-model separation
- Support for multiple state-of-the-art 3D detection models
- Efficient and GPU-accelerated processing

### 3. Model Selection

For this challenge, we implemented and compared multiple detection models:

- **PointPillar**: Fast and efficient detection with good accuracy for indoor scenes
- **SECOND**: Provides improved detection accuracy with sparse convolution
- **PV-RCNN**: Offers high-precision detection with point-voxel feature fusion

### 4. Custom Dataset Integration

We created a custom CIE dataset class that:
- Inherits from OpenPCDet's DatasetTemplate
- Provides proper point cloud feature encoding
- Handles the specific characteristics of the CIE space

### 5. Visualization

Our solution includes visualization tools that:
- Display the point cloud with detected objects
- Render 3D bounding boxes with class labels
- Allow for interactive exploration of the results

## Results

Our implementation successfully detects various objects in the CIE space, including:

- Chairs and tables
- Monitors and projection screens
- Projectors and TVs
- Computers and miscellaneous objects

The detection results are visualized with color-coded 3D bounding boxes, where each color represents a different object class.

## Implementation Details

### Directory Structure

```
├── configs/
│   ├── cie_dataset.yaml         # CIE dataset configuration
│   └── cie_pointpillar.yaml     # PointPillar model configuration for CIE
├── data/
│   └── cie/                     # CIE point cloud data
├── output/                      # Detection results
├── pcdet/                       # OpenPCDet library
└── tools/
    ├── cie_detection.py         # Main detection script
    ├── merge_pcd_files.py       # Utility to merge PCD files
    └── visualize_results.py     # Visualization tool
```

### Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Merge PCD Files**:
   ```bash
   python tools/merge_pcd_files.py --input_dir data/cie --output_path data/cie/merged_cie.pcd
   ```

3. **Run Object Detection**:
   ```bash
   python tools/cie_detection.py --cfg_file configs/cie_pointpillar.yaml --ckpt latest
   ```

4. **Visualize Results**:
   ```bash
   python tools/visualize_results.py --pcd_path data/cie/merged_cie.pcd --result_path output/result.pkl
   ```

## Future Work

Our solution could be further improved with:

1. Fine-tuning models specifically for the CIE environment
2. Implementing real-time detection for interactive applications
3. Adding semantic segmentation for more detailed scene understanding
4. Integrating with VR/AR applications for enhanced immersive experiences

## Team

This project was developed for the ICDS Challenge at HackPSU by our team.

## Acknowledgements

- Penn State Center for Immersive Experiences for providing the point cloud data
- OpenPCDet development team for their excellent 3D detection framework
- Open3D library for point cloud processing and visualization
