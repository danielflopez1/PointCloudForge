# Point Cloud Scene Generator

This Python script generates synthetic indoor scenes using point cloud objects. It's designed to create datasets for 3D scene understanding and object recognition tasks.

## Features

- Load point cloud objects from a directory structure
- Distribute objects across multiple areas and rooms
- Place objects in 3D scenes with collision detection
- Add floors, ceilings, and walls to each room
- Apply rotations to the scenes
- Save generated scenes as point cloud files
- Multi-processing support for faster execution

## Requirements

- Python 3.6+
- NumPy
- Open3D
- tqdm

## Installation

1. Clone this repository or download the script.
2. Install the required packages:

	pip install numpy open3d tqdm


## Usage

1. Prepare your input point cloud objects in a directory structure where each subdirectory represents a class.
2. Modify the script parameters in the `__main__` section:
   - `point_density`: Number of points to sample for planes
   - `num_areas`: Number of areas to generate
   - `rooms_per_area`: Number of rooms per area
   - `max_objects_per_scene`: Maximum number of objects per room
   - `dataset_names`: List of dataset names to process
   - `base_output_folder`: Base path for output datasets
   - `folder_path`: Path to input point cloud objects

3. Run the script:


## Output

The script generates a dataset with the following structure:

```plaintext
base_output_folder/
├── dataset_name_1/
│   ├── Area_1/
│   │   ├── room_1/
│   │   │   ├── Annotations/
│   │   │   │   ├── object_class_1.txt
│   │   │   │   ├── object_class_2.txt
│   │   │   │   ├── floor_1.txt
│   │   │   │   ├── wall_1.txt
│   │   │   │   └── ceiling_1.txt
│   │   │   └── room_1.txt
│   │   ├── room_2/
│   │   ├── ...
│   │   └── Area_1_alignmentAngle.txt
│   ├── Area_2/
│   └── ...
├── dataset_name_2/
└── ...
```


Each room contains individual object files in the Annotations folder and a combined room file with all objects and structural elements.

## Customization

- Adjust the `num_stories` and `story_height` parameters in the `place_objects_in_scene` function to change the vertical layout of rooms.
- Modify the `add_planes` function to customize the appearance of floors, walls, and ceilings.
- Change the rotation angles in the `save_scenes` function to alter the orientation of rooms.

## License

This software is licensed for educational and research purposes only. It may not be used for commercial purposes without explicit permission from the author(s).


## Contact

dflopezmorales@uwaterloo.ca
