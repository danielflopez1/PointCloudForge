import os
import sys
import numpy as np
import open3d as o3d
from collections import defaultdict
from tqdm import tqdm
import random
import shutil
import multiprocessing
from functools import partial
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PointCloudObject:
    def __init__(self, points, point_density, class_count, class_name):
        self.points = points
        self.point_density = point_density
        self.class_count = class_count
        self.class_name = class_name
        self.bounds = self.calculate_bounds()
        self.weight = self.calculate_weight()
        self.occurrences = 0

    def calculate_bounds(self):
        min_bounds = np.min(self.points[:, :3], axis=0)
        max_bounds = np.max(self.points[:, :3], axis=0)
        return min_bounds, max_bounds

    def get_dimensions(self):
        min_bounds, max_bounds = self.bounds
        return max_bounds - min_bounds

    def calculate_weight(self):
        class_count = max(self.class_count, 1)
        return len(self.points) / self.point_density / class_count

    def adjust_weight(self, placed):
        if placed:
            self.weight *= 0.9
        else:
            self.weight *= 1.1

    def increment_occurrence(self):
        self.occurrences += 1


def get_class_counts(folder_path):
    class_counts = defaultdict(int)
    for root, _, files in os.walk(folder_path):
        class_name = os.path.basename(root)
        txt_files = [file for file in files if file.endswith('.txt')]
        if class_name != '':
            class_counts[class_name] += len(txt_files)
    return class_counts


def load_point_cloud_file(args):
    file_path, point_density, class_counts = args
    try:
        points = np.loadtxt(file_path)
        if points.size == 0:
            return None
        class_name = os.path.basename(os.path.dirname(file_path))
        class_count = class_counts.get(class_name, 1)
        return PointCloudObject(points, point_density, class_count, class_name)
    except Exception as e:
        return None


def load_point_clouds_from_folder(folder_path, point_density, class_counts):
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
        args_list = [(file_path, point_density, class_counts) for file_path in file_paths]
        for result in tqdm(pool.imap_unordered(load_point_cloud_file, args_list), total=len(file_paths), desc="Loading Point Clouds"):
            if result is not None:
                point_clouds.append(result)
        pool.close()
        pool.join()

    logger.info(f"Successfully loaded {len(point_clouds)} point cloud objects.")

    return point_clouds


def distribute_objects_across_scenes(point_clouds, num_areas, rooms_per_area, max_objects_per_scene):
    if not point_clouds:
        return []

    total_rooms = num_areas * rooms_per_area
    scenes = [[] for _ in range(num_areas)]
    weights = np.array([obj.weight for obj in point_clouds], dtype=np.float64)

    if weights.sum() == 0:
        return []

    for room_index in tqdm(range(total_rooms), desc="Distributing Objects"):
        normalized_weights = weights / weights.sum()
        normalized_weights /= normalized_weights.sum()

        num_objects_to_select = min(max_objects_per_scene, len(point_clouds))
        if num_objects_to_select == 0:
            continue

        selected_indices = np.random.choice(len(point_clouds),
                                            size=num_objects_to_select,
                                            replace=False, p=normalized_weights)
        room_objects = [point_clouds[i] for i in selected_indices]

        for idx in selected_indices:
            point_clouds[idx].adjust_weight(placed=True)
            point_clouds[idx].increment_occurrence()
            weights[idx] = point_clouds[idx].weight

        non_selected_indices = np.setdiff1d(np.arange(len(point_clouds)), selected_indices)
        for idx in non_selected_indices:
            point_clouds[idx].adjust_weight(placed=False)
            weights[idx] = point_clouds[idx].weight

        area_index = room_index // rooms_per_area
        scenes[area_index].append(room_objects)

    return scenes


def place_objects_in_scene(
    room_objects,
    max_objects,
    num_stories=3,
    story_height=3,
    padding=0.5,
    room_width=10.0,
    room_length=10.0,
    max_attempts=1000
):
    placed_objects = []
    total_objects = min(max_objects, len(room_objects))
    objects_per_story = total_objects // num_stories
    extra_objects = total_objects % num_stories
    object_idx = 0

    for story in range(num_stories):
        current_z = story * story_height
        num_objects_this_story = objects_per_story + (1 if story < extra_objects else 0)
        story_placed_bboxes = []

        for _ in range(num_objects_this_story):
            if object_idx >= len(room_objects):
                break

            point_cloud = room_objects[object_idx]
            dimensions = point_cloud.get_dimensions() + padding
            placed = False
            attempt = 0

            while not placed and attempt < max_attempts:
                x_pos = random.uniform(0, room_width - dimensions[0])
                y_pos = random.uniform(0, room_length - dimensions[1])
                position = np.array([x_pos, y_pos, current_z])

                min_bounds = position
                max_bounds = position + dimensions
                obj_bbox = (min_bounds[:2], max_bounds[:2])

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
                pass

            object_idx += 1

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
    scene_objects, base_path, area_index, room_index, point_density, noise_level, num_stories, story_height, rotation_angle, axis = args
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
        if obj.points.shape[1] == 6:
            colors = obj.points[:, 3:]
        else:
            colors = np.zeros((rotated_points.shape[0], 3))

        room_points.append(rotated_points)
        room_colors.append(colors)

        points_with_color = np.hstack((rotated_points, colors))
        object_file_path = os.path.join(annotation_path, f'{obj.class_name}_{class_occurrence}.txt')
        np.savetxt(object_file_path, points_with_color, fmt='%f')

    room_points = np.concatenate(room_points, axis=0)
    room_colors = np.concatenate(room_colors, axis=0)

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


def save_scenes(scenes, base_path, point_density, max_objects_per_scene, noise_level):
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
            scene_objects = place_objects_in_scene(room_objects, max_objects_per_scene, num_stories, story_height)

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
        pool.close()
        pool.join()


def delete_output_folder(base_path):
    if os.path.exists(base_path):
        shutil.rmtree(base_path)

if __name__ == "__main__":
    point_density = 5000
    num_areas = 23
    rooms_per_area = 45
    max_objects_per_scene = 30

    dataset_names = [f'subsample_{point_density}_0cm',f'subsample_{point_density}_2cm',f'subsample_{point_density}_5cm']

    base_output_folder = 'path/to/dataset/base'

    for name in dataset_names:
        folder_path = f'path/to/pointclouds/ObjectsTXT/{name}'
        base_path = os.path.join(base_output_folder, name)

        delete_output_folder(base_path)

        class_counts = get_class_counts(folder_path)
        point_clouds = load_point_clouds_from_folder(folder_path, point_density, class_counts)

        if not point_clouds:
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
            sys.exit(1)

        save_scenes(scenes, base_path, point_density, max_objects_per_scene, noise_level)

    logger.info("Scene generation completed.")
