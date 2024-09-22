
# PointCloudForge - Object Standardizer

### Overview

**PointCloudForge** is a tool designed to generate, process, and visualize point cloud datasets. It provides customizable settings for lighting and rendering to simulate various environmental conditions, allowing for the creation of high-quality, tailored datasets for 3D models. This repository includes a module for object standardization using the Open3D library, which is key in rendering and visualizing point clouds.

### Features

- **Multiple Lighting Profiles**: Simulate various environmental conditions using predefined profiles (e.g., bright day, cloudy day).
- **Point Cloud Visualization**: Leverage Open3D to visualize and render point cloud data with flexible controls.
- **Customizable Settings**: Modify IBL (Image Based Lighting) intensity, sun intensity, and direction to match specific use cases.
- **Cross-Platform**: Supports both macOS and other operating systems for seamless 3D data processing.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/PointCloudForge.git
   cd PointCloudForge
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Requirements

- **Python 3.8+**
- **Open3D**
- **NumPy**

Ensure the following libraries are installed:

```bash
pip install open3d numpy
```

### Usage

To use the object standardizer, run the `ObjectStandarizer.py` script. This script provides a customizable environment to visualize and adjust point cloud data.

```bash
python ObjectStandarizer.py
```

You can choose between different lighting profiles, adjust sun direction, and manipulate point cloud data in real-time.

### Lighting Profiles

- **Bright day with sun at +Y**: High-intensity lighting simulating a bright day with the sun overhead.
- **Cloudy day**: Low-intensity lighting to mimic overcast weather.
- **Custom profiles**: Create your own lighting conditions by adjusting parameters like sun intensity and direction.

### Contributing

Contributions are welcome! Feel free to submit pull requests or report issues.

### License

This project is licensed under the MIT License. See the `LICENSE` file for details.
