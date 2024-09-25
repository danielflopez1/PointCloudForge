
# PointCloudForge

Welcome to **PointCloudForge**, a powerful tool for generating customizable point cloud datasets. This tool is designed to assist in the creation of complex, diverse point cloud data that can be used for a variety of purposes, including 3D modeling, simulations, and machine learning tasks.

## Features

- **Customizable Point Cloud Generation**: Generate diverse and complex point cloud datasets tailored to your specific needs.
- **Support for 3D Modeling and Simulations**: Useful for 3D rendering, object recognition, and environmental simulations.
- **Efficient for Machine Learning**: Balanced dataset generation ensures optimal training data for deep learning models.
- **Integration with Stable Diffusion, InstantMesh, and Open3D**: The tool uses procedural generation techniques from state-of-the-art models to produce realistic and detailed point clouds.

## Object Generation

Object generation within PointCloudForge is based on three key technologies:

1. **Stable Diffusion Medium**: A model from Stability AI, which allows for the generation of medium-sized 3D objects. You can find more information about this model on the official HuggingFace space: [Stable Diffusion Medium](https://huggingface.co/spaces/stabilityai/stable-diffusion-3-medium).
   
2. **InstantMesh**: Developed by TencentARC, InstantMesh is used for creating high-quality, procedurally generated meshes that serve as the foundation for point cloud data. Explore this model on HuggingFace here: [InstantMesh](https://huggingface.co/spaces/TencentARC/InstantMesh).

## Dataset Generation
Refer to the [Dataset Generator](https://github.com/danielflopez1/PointCloudForge/tree/main/DatasetGeneration)

## Getting Started

### Prerequisites

Before using PointCloudForge, make sure you have the following dependencies installed:

- Python 3.8+
- Required Python libraries: 
  - `numpy`
  - `multiprocessing`
  - `open3d`
  - `tqdm`
  - `shutil`
  - `re`
  - `logging`

To install the required dependencies, run:

```bash
pip install numpy open3d tqdm
```

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/PointCloudForge.git
   ```

2. Navigate to the project directory:

   ```bash
   cd PointCloudForge
   ```

3. Run the point cloud generation script:

   ```bash
   python pointcloudforge.py
   ```

### Usage

PointCloudForge allows you to generate customized point cloud datasets. You can adjust parameters such as:

- **Number of objects per scene**
- **Number of scenes**
- **Point density**
- **Noise levels**

To start generating point clouds, edit the configuration variables in the main script or pass them as arguments:

```bash
python pointcloudforge.py --num_objects 30 --num_scenes 100 --point_density 5000
```

### Example

To generate 50 scenes with a maximum of 20 objects per scene and a point density of 3000, run:

```bash
python pointcloudforge.py --num_scenes 50 --max_objects_per_scene 20 --point_density 3000
```

## Output

Generated point cloud datasets will be saved in the specified output directory. The output includes:

- **Scene Files**: Each scene will have a corresponding `.txt` file containing point cloud data.
- **Metadata**: Information about the objects, point distribution, and noise levels used in the generation process.

## Logging and Verification

The script includes built-in logging and verification functionalities:

- **Logging**: Detailed logs provide insights into the dataset generation process. Logs include information about errors, object distribution, and point balance across scenes.
- **Verification**: After generation, datasets are verified for format consistency, ensuring point cloud files conform to expected standards.

## Contributing

We welcome contributions to improve PointCloudForge. If you want to contribute:

1. Fork the repository.
2. Create a new branch.
3. Make your changes and submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgments

PointCloudForge is built upon advanced technologies such as:

- [Stable Diffusion Medium](https://huggingface.co/spaces/stabilityai/stable-diffusion-3-medium) for 3D object generation.
- [InstantMesh](https://huggingface.co/spaces/TencentARC/InstantMesh) for procedural mesh generation.
- [Open3D](http://www.open3d.org/) for efficient point cloud processing and manipulation.

Special thanks to the developers of these models and libraries for their contributions to the field of 3D data generation.

---

For more information, visit the official documentation or reach out to the community via GitHub Issues.
