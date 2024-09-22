
# PointCloudForge - Object Standardizer

### Overview

**PointCloudForge** is a tool designed to generate, process, and visualize point cloud datasets. The **Object Standardizer** module provides a flexible environment to manipulate, render, and standardize point cloud data. With built-in lighting profiles and rendering controls, users can simulate real-world conditions and create clean, standardized datasets. This project is built on the **Open3D** library, which provides advanced 3D visualization capabilities.

### Features

- **Multiple Lighting Profiles**: Simulate different lighting environments using predefined profiles, such as bright days or cloudy weather.
- **Customizable Visualizations**: Modify point cloud rendering settings like sun intensity, IBL intensity, and sun direction for more realistic visualizations.
- **Cross-Platform Compatibility**: Compatible with macOS and other operating systems.
- **Interactive Point Cloud Viewer**: Open3D's GUI allows for real-time interaction with point clouds, including zoom, pan, and rotation.
- **Point Cloud Normalization**: Normalize 3D objects by standardizing their dimensions, orientation, and appearance.
- **Efficient Processing**: Load, transform, and visualize large point cloud datasets with efficient Open3D rendering.

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

To use the object standardizer, run the `ObjectStandarizer.py` script. The script provides the following key functionalities:

1. **Lighting Profiles**: Choose between several pre-defined lighting profiles, including:
   - **Bright Day with Sun at +Y**: High IBL intensity simulating direct sunlight.
   - **Cloudy Day**: Lower intensity, mimicking diffuse lighting.
   - **Customizable Profiles**: Adjust parameters like IBL intensity, sun intensity, and sun direction to create custom environments.
   
2. **Rendering Modes**: Switch between rendering modes to highlight different aspects of the point cloud:
   - **Unlit Mode**: Visualizes the point cloud without lighting effects.
   - **Lit Mode**: Applies realistic lighting to the point cloud.
   - **Normals Mode**: Visualizes the normals of each point, useful for debugging and mesh refinement.
   - **Depth Mode**: Visualizes the depth of the point cloud for better spatial understanding.

3. **Interactive Viewer**: Use Open3D's interactive GUI for real-time visualization of point clouds:
   - Rotate, pan, and zoom around your point cloud.
   - Adjust lighting profiles on the fly using the GUI controls.
   - Supports rendering both 3D models and raw point clouds.

4. **Cross-Platform**: The module works on macOS, Windows, and Linux, utilizing platform-specific optimizations for a smooth experience.

### Lighting Profiles

- **Bright day with sun at +Y**: High-intensity lighting simulating a bright day with the sun overhead.
- **Bright day with sun at -Y**: Simulates sunlight from below the object.
- **Cloudy day (no direct sun)**: Simulates overcast conditions with diffuse lighting.
- **Custom profiles**: Create your own lighting conditions by adjusting parameters such as sun intensity and direction.

### Contributing

Contributions are welcome! Feel free to submit pull requests or report issues.

### License

This project is licensed under the MIT License. See the `LICENSE` file for details.
