import os
import sys
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import multiprocessing
import logging
import open3d as o3d
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PointCloudObject:
    def __init__(self, points, class_name):
        self.points = points
        self.class_name = class_name
        self.num_points = points.shape[0]
        self.bounds = self.calculate_bounds()

    def calculate_bounds(self):
        min_bounds = np.min(self.points[:, :3], axis=0)
        max_bounds = np.max(self.points[:, :3], axis=0)
        return min_bounds, max_bounds

    def get_dimensions(self):
        min_bounds, max_bounds = self.bounds
        return max_bounds - min_bounds


def load_point_cloud_file(file_path):
    try:
        points = np.loadtxt(file_path)
        if points.size == 0:
            return None
        class_name = os.path.basename(os.path.dirname(file_path))
        return PointCloudObject(points, class_name)
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        return None


def load_point_clouds_from_folder(folder_path):
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.txt'):
                file_path = os.path.join(root, file_name)
                file_paths.append(file_path)

    logger.info(f"Found {len(file_paths)} point cloud files in {folder_path}.")

    if len(file_paths) == 0:
        return []

    point_clouds = []
    with multiprocessing.Pool() as pool:
        for result in tqdm(pool.imap_unordered(load_point_cloud_file, file_paths), total=len(file_paths),
                           desc="Loading Point Clouds"):
            if result is not None:
                point_clouds.append(result)

    logger.info(f"Successfully loaded {len(point_clouds)} point cloud objects.")
    return point_clouds


def distribute_objects_across_scenes(point_clouds, num_areas, rooms_per_area, max_objects_per_scene):
    if not point_clouds:
        return []

    # Group objects by class and calculate average points per object per class
    class_to_objects = defaultdict(list)
    class_to_total_points = defaultdict(int)
    for obj in point_clouds:
        class_to_objects[obj.class_name].append(obj)
        class_to_total_points[obj.class_name] += obj.num_points

    classes = list(class_to_objects.keys())
    num_classes = len(classes)
    total_scenes = num_areas * rooms_per_area
    total_objects_in_dataset = total_scenes * max_objects_per_scene

    # Calculate average points per object across all objects
    total_points_all_objects = sum(obj.num_points for obj in point_clouds)
    average_points_per_object_all = total_points_all_objects / len(point_clouds)

    # Estimate total desired points in dataset
    total_desired_points_in_dataset = total_objects_in_dataset * average_points_per_object_all

    # Desired points per class
    desired_points_per_class = total_desired_points_in_dataset / num_classes

    # Calculate average points per object per class
    class_to_avg_points = {}
    for cls in classes:
        total_points = class_to_total_points[cls]
        num_objects = len(class_to_objects[cls])
        class_to_avg_points[cls] = total_points / num_objects

    # Compute required number of objects per class
    class_to_required_objects = {}
    total_required_objects = 0
    for cls in classes:
        avg_points = class_to_avg_points[cls]
        required_objects = int(np.ceil(desired_points_per_class / avg_points))
        class_to_required_objects[cls] = required_objects
        total_required_objects += required_objects

    # Adjust required objects per class if total exceeds capacity
    max_total_objects = total_scenes * max_objects_per_scene
    if total_required_objects > max_total_objects:
        scaling_factor = max_total_objects / total_required_objects
        for cls in classes:
            class_to_required_objects[cls] = int(class_to_required_objects[cls] * scaling_factor)
        total_required_objects = sum(class_to_required_objects.values())

    # Distribute objects across scenes
    scenes = [[] for _ in range(num_areas)]
    class_counts = defaultdict(int)  # Total counts per class across all areas
    class_points = defaultdict(int)  # Total points per class across all areas

    # Prepare a list of all scenes
    all_scenes = []
    for area_index in range(num_areas):
        for scene_index in range(rooms_per_area):
            all_scenes.append((area_index, scene_index))

    # Shuffle scenes to distribute classes evenly
    random.shuffle(all_scenes)

    # Create a pool of objects per class (allow sampling with replacement)
    class_to_object_pool = {}
    for cls in classes:
        class_to_object_pool[cls] = class_to_objects[cls]

    # For each class, distribute objects across scenes
    class_to_scene_objects = defaultdict(list)
    for cls in classes:
        required_objects = class_to_required_objects[cls]
        available_objects = class_to_objects[cls]
        num_available = len(available_objects)
        objects = []

        # Sample objects with replacement if needed
        if required_objects <= num_available:
            objects = random.sample(available_objects, required_objects)
        else:
            # Sample all available objects and then sample additional with replacement
            objects = available_objects.copy()
            additional_needed = required_objects - num_available
            objects.extend(random.choices(available_objects, k=additional_needed))

        # Assign objects to scenes
        scene_indices = np.linspace(0, len(all_scenes) - 1, required_objects, dtype=int)
        for idx, obj in zip(scene_indices, objects):
            area_idx, scene_idx = all_scenes[idx]
            class_to_scene_objects[(area_idx, scene_idx)].append(obj)
            class_counts[cls] += 1
            class_points[cls] += obj.num_points

    # Assemble scenes
    for area_index in range(num_areas):
        for scene_index in range(rooms_per_area):
            key = (area_index, scene_index)
            scene_objects = class_to_scene_objects.get(key, [])
            # If the scene has fewer objects than max_objects_per_scene, fill with random objects
            if len(scene_objects) < max_objects_per_scene:
                num_additional = max_objects_per_scene - len(scene_objects)
                # Collect all objects from all classes
                all_objects = [obj for objs in class_to_objects.values() for obj in objs]
                additional_objects = random.choices(all_objects, k=num_additional)
                scene_objects.extend(additional_objects)
            else:
                scene_objects = scene_objects[:max_objects_per_scene]

            scenes[area_index].append(scene_objects)

    # Print total class counts and points
    logger.info("Total class counts and points across all areas:")
    for cls in classes:
        logger.info(f"  {cls}: {class_counts[cls]} objects, {class_points[cls]} points")

    return scenes


def place_objects_in_scene(
    room_objects,
    num_stories=3,
    story_height=3,
    padding=0.1,
    room_width=20.0,
    room_length=20.0,
    max_attempts=1000
):
    placed_objects = []
    random.shuffle(room_objects)  # Shuffle objects to introduce randomness
    total_objects = len(room_objects)
    objects_per_story = total_objects // num_stories
    extra_objects = total_objects % num_stories
    object_idx = 0

    for story in range(num_stories):
        current_z = story * story_height
        num_objects_this_story = objects_per_story + (1 if story < extra_objects else 0)
        story_placed_bboxes = []

        while num_objects_this_story > 0 and object_idx < len(room_objects):
            point_cloud = room_objects[object_idx]
            dimensions = point_cloud.get_dimensions() + padding

            # Check if object can fit
            if dimensions[0] > room_width or dimensions[1] > room_length:
                logger.warning(f"Object {point_cloud.class_name} is too large to fit in the room. Skipping.")
                object_idx += 1
                num_objects_this_story -= 1
                continue

            placed = False
            attempt = 0

            while not placed and attempt < max_attempts:
                x_pos = random.uniform(0, room_width - dimensions[0])
                y_pos = random.uniform(0, room_length - dimensions[1])
                position = np.array([x_pos, y_pos, current_z])

                min_bounds = position
                max_bounds = position + dimensions

                collision = False
                for placed_min, placed_max in story_placed_bboxes:
                    if (
                        min_bounds[0] < placed_max[0] and max_bounds[0] > placed_min[0] and
                        min_bounds[1] < placed_max[1] and max_bounds[1] > placed_min[1]
                    ):
                        collision = True
                        break

                if not collision:
                    placed_objects.append((point_cloud, position))
                    story_placed_bboxes.append((min_bounds[:2], max_bounds[:2]))
                    placed = True
                else:
                    attempt += 1

            if not placed:
                logger.warning(f"Failed to place object {point_cloud.class_name} after {max_attempts} attempts.")

            object_idx += 1
            num_objects_this_story -= 1

    return placed_objects


def create_plane(vertices, color):
    plane = o3d.geometry.TriangleMesh()
    plane.vertices = o3d.utility.Vector3dVector(vertices)
    plane.triangles = o3d.utility.Vector3iVector([
        [0, 1, 2], [2, 3, 0]
    ])
    plane.compute_vertex_normals()
    plane.paint_uniform_color(color)
    return plane


def mesh_to_point_cloud(mesh, num_points=1000):
    sampled_points = mesh.sample_points_uniformly(number_of_points=num_points)
    colors = np.asarray(sampled_points.colors)
    if colors.size == 0:
        colors = np.ones((num_points, 3)) * 0.5
    else:
        colors += np.random.uniform(-0.05, 0.05, colors.shape)
        colors = np.clip(colors, 0, 1)
    sampled_points.colors = o3d.utility.Vector3dVector(colors)
    return sampled_points


def add_planes(annotation_path, room_points, point_density, noise_level, num_stories, story_height):
    if room_points.size == 0:
        # If there are no objects, define default room bounds
        min_bound = np.array([0, 0, 0])
        max_bound = np.array([20.0, 20.0, num_stories * story_height])
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(room_points)
        aabb = pcd.get_axis_aligned_bounding_box()
        min_bound = aabb.min_bound
        max_bound = aabb.max_bound

    floors = []

    for story in range(num_stories):
        z_offset = story * story_height

        floor_vertices = [
            [min_bound[0], min_bound[1], min_bound[2] + z_offset],
            [max_bound[0], min_bound[1], min_bound[2] + z_offset],
            [max_bound[0], max_bound[1], min_bound[2] + z_offset],
            [min_bound[0], max_bound[1], min_bound[2] + z_offset]
        ]
        floor_color = [204, 204, 204]
        floors.append(create_plane(floor_vertices, np.array(floor_color) / 255))

    ceiling_vertices = [
        [min_bound[0], min_bound[1], max_bound[2]],
        [max_bound[0], min_bound[1], max_bound[2]],
        [max_bound[0], max_bound[1], max_bound[2]],
        [min_bound[0], max_bound[1], max_bound[2]]
    ]

    wall1_vertices = [
        [min_bound[0], min_bound[1], min_bound[2]],
        [min_bound[0], max_bound[1], min_bound[2]],
        [min_bound[0], max_bound[1], max_bound[2]],
        [min_bound[0], min_bound[1], max_bound[2]]
    ]

    wall2_vertices = [
        [max_bound[0], min_bound[1], min_bound[2]],
        [max_bound[0], max_bound[1], min_bound[2]],
        [max_bound[0], max_bound[1], max_bound[2]],
        [max_bound[0], min_bound[1], max_bound[2]]
    ]

    wall3_vertices = [
        [min_bound[0], min_bound[1], min_bound[2]],
        [max_bound[0], min_bound[1], min_bound[2]],
        [max_bound[0], min_bound[1], max_bound[2]],
        [min_bound[0], min_bound[1], max_bound[2]]
    ]

    wall4_vertices = [
        [min_bound[0], max_bound[1], min_bound[2]],
        [max_bound[0], max_bound[1], min_bound[2]],
        [max_bound[0], max_bound[1], max_bound[2]],
        [min_bound[0], max_bound[1], max_bound[2]]
    ]

    ceiling_color = [230, 230, 230]
    wall1_color = [240, 235, 214]
    wall2_color = [217, 217, 217]
    wall3_color = [230, 230, 230]
    wall4_color = [204, 204, 204]

    def add_noise_to_color(color, noise_level):
        color = np.array(color)
        noisy_color = color + np.random.randint(-noise_level, noise_level + 1, color.shape)
        return np.clip(noisy_color, 0, 255) / 255.0

    ceiling = create_plane(ceiling_vertices, np.array(ceiling_color) / 255)
    wall1 = create_plane(wall1_vertices, add_noise_to_color(wall1_color, noise_level))
    wall2 = create_plane(wall2_vertices, add_noise_to_color(wall2_color, noise_level))
    wall3 = create_plane(wall3_vertices, add_noise_to_color(wall3_color, noise_level))
    wall4 = create_plane(wall4_vertices, add_noise_to_color(wall4_color, noise_level))

    floor_pcs = [mesh_to_point_cloud(floor, point_density) for floor in floors]
    ceiling_pc = mesh_to_point_cloud(ceiling, point_density)
    wall_pcs = [mesh_to_point_cloud(wall, point_density) for wall in [wall1, wall2, wall3, wall4]]

    return floor_pcs, ceiling_pc, wall_pcs


def generate_rotation_matrix(angle_deg, axis):
    angle_rad = np.radians(angle_deg)
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError(f"Invalid rotation axis: {axis}")
    return rotation_matrix


def save_scene(args):
    (
        scene_objects,
        base_path,
        area_index,
        room_index,
        point_density,
        noise_level,
        num_stories,
        story_height,
        rotation_angle,
        axis
    ) = args

    room_path = os.path.join(base_path, f'Area_{area_index + 1}', f'room_{room_index + 1}')
    annotation_path = os.path.join(room_path, 'Annotations')

    os.makedirs(annotation_path, exist_ok=True)

    room_points = []
    room_colors = []
    class_counter = defaultdict(int)

    rotation_matrix = generate_rotation_matrix(rotation_angle, axis)

    for obj, position in scene_objects:
        class_counter[obj.class_name] += 1
        class_occurrence = class_counter[obj.class_name]

        translated_points = obj.points[:, :3] + position
        rotated_points = np.dot(translated_points, rotation_matrix.T)
        if obj.points.shape[1] >= 6:
            colors = obj.points[:, 3:6]
        else:
            colors = np.zeros((rotated_points.shape[0], 3))

        room_points.append(rotated_points)
        room_colors.append(colors)

        points_with_color = np.hstack((rotated_points, colors))
        object_file_path = os.path.join(annotation_path, f'{obj.class_name}_{class_occurrence}.txt')
        np.savetxt(object_file_path, points_with_color, fmt='%f')

    # Handle empty room_points and room_colors
    if room_points:
        room_points = np.concatenate(room_points, axis=0)
        room_colors = np.concatenate(room_colors, axis=0)
    else:
        room_points = np.empty((0, 3))
        room_colors = np.empty((0, 3))

    floor_pcs, ceiling_pc, wall_pcs = add_planes(
        annotation_path, room_points, point_density, noise_level, num_stories, story_height)

    for idx, floor_pc in enumerate(floor_pcs):
        floor_points_with_color = np.hstack((np.asarray(floor_pc.points), np.asarray(floor_pc.colors) * 255))
        floor_file_path = os.path.join(annotation_path, f'floor_{idx + 1}.txt')
        np.savetxt(floor_file_path, floor_points_with_color, fmt='%f')

    for idx, wall_pc in enumerate(wall_pcs):
        wall_points_with_color = np.hstack((np.asarray(wall_pc.points), np.asarray(wall_pc.colors) * 255))
        wall_file_path = os.path.join(annotation_path, f'wall_{idx + 1}.txt')
        np.savetxt(wall_file_path, wall_points_with_color, fmt='%f')

    ceiling_points_with_color = np.hstack((np.asarray(ceiling_pc.points), np.asarray(ceiling_pc.colors) * 255))
    ceiling_file_path = os.path.join(annotation_path, f'ceiling_1.txt')
    np.savetxt(ceiling_file_path, ceiling_points_with_color, fmt='%f')

    all_planes_points = np.vstack([np.asarray(pc.points) for pc in floor_pcs + [ceiling_pc] + wall_pcs])
    all_planes_colors = np.vstack([np.asarray(pc.colors) * 255 for pc in floor_pcs + [ceiling_pc] + wall_pcs])

    room_points = np.concatenate([room_points, all_planes_points], axis=0)
    room_colors = np.concatenate([room_colors, all_planes_colors], axis=0)

    room_points_with_colors = np.hstack((room_points, room_colors))

    combined_room_file_path = os.path.join(room_path, f'room_{room_index + 1}.txt')
    np.savetxt(combined_room_file_path, room_points_with_colors, fmt='%f')


def save_alignment_angles(base_path, area_index, alignment_angles):
    alignment_file_path = os.path.join(base_path, f'Area_{area_index + 1}', f'Area_{area_index + 1}_alignmentAngle.txt')
    os.makedirs(os.path.dirname(alignment_file_path), exist_ok=True)
    with open(alignment_file_path, 'w') as f:
        f.write(f"## Global alignment angle per disjoint space in Area_{area_index + 1} ##\n")
        f.write("## Disjoint Space Name Global Alignment Angle ##\n")
        for room_index, alignment_angle in alignment_angles:
            f.write(f"room_{room_index} {alignment_angle}\n")


def save_scenes(scenes, base_path, point_density, noise_level):
    args_list = []
    for area_index, area in enumerate(scenes):
        alignment_angles = []
        for room_index, room_objects in enumerate(area):
            num_stories = random.randint(2, 3)
            story_height = 3
            rotation_angle = random.choice([0, 90, 180, 270])
            axis = 'z'
            alignment_angle = rotation_angle
            alignment_angles.append((room_index + 1, alignment_angle))
            scene_objects = place_objects_in_scene(
                room_objects,
                num_stories=num_stories,
                story_height=story_height,
                padding=0.1,
                room_width=20.0,
                room_length=20.0,
                max_attempts=1000
            )

            args = (
                scene_objects,
                base_path,
                area_index,
                room_index,
                point_density,
                noise_level,
                num_stories,
                story_height,
                rotation_angle,
                axis
            )
            args_list.append(args)

        save_alignment_angles(base_path, area_index, alignment_angles)

    with multiprocessing.Pool() as pool:
        list(tqdm(pool.imap_unordered(save_scene, args_list), total=len(args_list), desc="Saving Scenes"))


def delete_output_folder(base_path):
    if os.path.exists(base_path):
        shutil.rmtree(base_path)


if __name__ == "__main__":
    point_density = 5000
    num_areas = 2
    rooms_per_area = 45
    max_objects_per_scene = 30

    dataset_names = [
        f'subsample_{point_density}_0cm'
    ]

    base_output_folder = '/home/daniel/PycharmProjects/DatasetGenerator/S3DIS_Scenes3/'

    for name in dataset_names:
        folder_path = f'/home/daniel/PycharmProjects/DatasetGenerator/ObjectsTXT/{name}'
        base_path = os.path.join(base_output_folder, name)

        point_clouds = load_point_clouds_from_folder(folder_path)

        if not point_clouds:
            logger.error("No point clouds were loaded. Exiting.")
            sys.exit(1)

        if '0cm' in name:
            noise_level = 0
        elif '2cm' in name:
            noise_level = 2
        elif '5cm' in name:
            noise_level = 5
        else:
            noise_level = 0

        scenes = distribute_objects_across_scenes(point_clouds, num_areas, rooms_per_area, max_objects_per_scene)

        if not scenes:
            logger.error("No scenes were generated. Exiting.")
            sys.exit(1)

        save_scenes(scenes, base_path, point_density, noise_level)

    logger.info("Scene generation completed.")
