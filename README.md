# Point Cloud Aligner

A Qt-based interactive tool for manually aligning point clouds with visual feedback and automatic refinement using ColorICP algorithm.

## Features

- **Interactive Point Cloud Visualization**: Load and visualize multiple point clouds in 3D
- **Manual Alignment Controls**: Fine-tune transformations with 6DOF controls (translation and rotation)
- **Reference Cloud Selection**: Choose any cloud as the reference frame for alignment
- **ColorICP Refinement**: Automatic alignment refinement using color information
- **Transform Visualization**: Visual arrows showing transformations between clouds
- **Export Capabilities**: Export transformation matrices and 6DOF values
- **Visibility Controls**: Toggle individual cloud visibility
- **Color Modes**: View clouds with original colors or unique colors per cloud

## Prerequisites

- CMake 3.10 or higher
- Qt5 with VTK support
- PCL (Point Cloud Library)
- OpenCV
- VTK
- Eigen3

## Installation

### Ubuntu/Debian

```bash
# Install dependencies
sudo apt update
sudo apt install cmake build-essential
sudo apt install libpcl-dev libopencv-dev libvtk7-dev
sudo apt install qt5-default libqt5opengl5-dev
sudo apt install libeigen3-dev

# Clone the repository
git clone https://github.com/ShengBin-101/pointcloud-aligner.git
cd pointcloud-aligner

# Build the project
mkdir build && cd build
cmake ..
make -j8
```

### Running the Application

```bash
cd build
./pointcloud_aligner
```

## Usage

### 1. Loading Point Clouds

1. Click **"Load Point Cloud"** button
2. Select PCD files to load
3. Loaded clouds appear in the main viewer and control lists

### 2. Cloud Selection

- **Reference/Fixed Cloud**: Select the cloud that will remain stationary (reference frame)
- **Moving Cloud**: Select the cloud you want to transform and align

### 3. Manual Alignment

- **Step Size**: Adjust the increment for each transformation step
- **Translation Controls**: Use +/-X, +/-Y, +/-Z buttons for translation
- **Rotation Controls**: Use +/-RX, +/-RY, +/-RZ buttons for rotation
- **Transform Display**: View current transformation matrix and DOF values

### 4. Automatic Refinement

- Click **"Refine with ColorICP"** to automatically improve alignment using color information
- The algorithm uses both geometric and color features for better accuracy

### 5. Transform Visualization

- Click **"Visualize Transforms"** to show arrows from reference cloud to other clouds
- Arrows display direction, distance, and cloud names
- Updates automatically when transformations change

### 6. Export Transforms

- Click **"Export Transforms"** to get transformation matrices
- Data appears in terminal and copyable popup dialog
- Includes 4x4 extrinsic matrices and 6DOF values relative to reference cloud

### 7. Display Options

- **Original Point Colors**: Use RGB values from point cloud files
- **Unique Color per Cloud**: Assign different colors to distinguish clouds
- **Visibility Toggle**: Use eye icons to show/hide individual clouds
- **Reference Frame**: Toggle coordinate axes display

## Controls Summary

| Control | Function |
|---------|----------|
| +/-X, +/-Y, +/-Z | Translation along axes |
| +/-RX, +/-RY, +/-RZ | Rotation around axes |
| Step Size | Adjustment increment |
| ColorICP | Automatic alignment refinement |
| Visualize Transforms | Show transformation arrows |
| Export Transforms | Output transformation data |

## File Formats

- **Input**: PCD (Point Cloud Data) files with RGB information
- **Output**: Transformation matrices in 4x4 format and 6DOF parameters


