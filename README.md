# Scan-To-HBIM: Sequential BPA–Poisson Reconstruction Framework

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18635433.svg)](https://doi.org/10.5281/zenodo.18635433)

## Abstract
This repository presents a reproducible Python implementation of a semi-automated Scan-to-HBIM workflow for converting segmented point cloud data into geometrically consistent surface meshes suitable for architectural and structural modeling.
The framework integrates adaptive point cloud preprocessing with a sequential surface reconstruction strategy combining the Ball Pivoting Algorithm (BPA) and Poisson surface reconstruction. The method incorporates automatic parameter estimation, mesh quality evaluation, connected component filtering, and geometric cleaning to ensure stable and reproducible results.

**The implementation is designed for research reproducibility and academic evaluation.**

**1. Introduction**

Scan-to-HBIM workflows require robust surface reconstruction from segmented laser-scanned point clouds. However, several challenges commonly arise:
Noise and irregular density
Fragmented surfaces
Open boundaries
Small disconnected mesh components
These issues often lead to unstable or incomplete meshing results.

**This framework addresses these challenges through:**

Adaptive point cloud preprocessing

Automatic normal estimation

Sequential BPA reconstruction

Conditional Poisson reconstruction fallback

Mesh component filtering and geometric cleaning


**2. Workflow Overview**

The implemented pipeline consists of two main stages:

**2.1 Point Cloud Preprocessing**

Adaptive voxel-based downsampling
Local spacing estimation
Feature-aware keypoint preservation
Statistical outlier removal
Automatic normal estimation

**Script:**

pointcloud_preprocessing.py

**2.2 Sequential Surface Reconstruction**

Automatic BPA radius estimation

Mesh quality evaluation

Conditional fallback to Poisson reconstruct

Connected component filtering

Bounding box cropping

Final mesh cleanup


**Script:**

sequential_bpa_poisson_meshing.py

**3. Installation**

3.1 Clone Repository
git clone https://github.com/nithinbattu0/Scan-To-HBIM.git
cd Scan-To-HBIM

**3.2 Create Virtual Environment (Recommended)**

python -m venv venv

Windows:
venv\Scripts\activate

Linux / macOS:

source venv/bin/activate

**3.3 Install Dependencies**

pip install -r requirements.txt

**4. Usage**

**4.1 Preprocessing Stage**

python pointcloud_preprocessing.py --input sample_segmented_input.ply --output preprocessed.ply

**4.2 Surface Reconstruction Stage**

python sequential_bpa_poisson_meshing.py --input preprocessed.ply --output final_mesh.ply

**5. Reproducibility**

To ensure academic reproducibility:
A sample segmented dataset (sample_segmented_input.ply) is included.
All algorithmic parameters are automatically derived from geometric properties of the input data.
No manual tuning is required for standard segmented datasets.
The complete workflow can be reproduced using only the provided repository files.

**6. Dependencies**

Python 3.8+
Open3D
NumPy

**All required packages are listed in:**

requirements.txt

**7. Applications**

Heritage digitization

Structural documentation

Scan-to-CAD workflows

HBIM research

Academic evaluation


**8. Limitations**

Extremely sparse datasets may require parameter adjustment.
Very large datasets may require memory optimization.
The pipeline assumes pre-segmented input data.


## 9. Citation and DOI

The archived and DOI-registered version of this software is available via Zenodo:

https://doi.org/10.5281/zenodo.18635433

If you use this software in academic work, please cite:

Battu Nithin. (2026). *Scan-To-HBIM: Sequential BPA–Poisson Scan-to-HBIM Pipeline (Version 1.0.1)*. Zenodo. https://doi.org/10.5281/zenodo.18635433


**10. License**

This project is licensed under the MIT License.

**Author**

Battu Nithin
M.Tech – Construction Technology and Management
Scan-to-HBIM Research

**GitHub:** https://github.com/nithinbattu0
