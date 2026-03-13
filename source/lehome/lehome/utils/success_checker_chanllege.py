import math
import numpy as np
import torch
from lehome.utils.logger import get_logger

logger = get_logger(__name__)


def step_interval(interval=50):
    """Factory function: creates a customizable step interval decorator"""

    def decorator(func):
        call_count = 0

        def wrapper(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count % interval == 0:
                return func(*args, **kwargs)
            else:
                # Return False for skipped steps (maintains backward compatibility)
                # For success_checker_garment_fold, this will be handled in _check_success
                return False

        return wrapper

    return decorator


def calculate_distance(point_a, point_b):
    # Calculate distance
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    return np.linalg.norm(point_a - point_b)


def get_object_particle_position(particle_object, index_list):
    try:
        transformed_mesh_points, _, _, _ = particle_object.get_current_mesh_points()
    except Exception as e1:
        try:
            logger.error(f"Error in get_object_particle_position: {e1}")
            transformed_mesh_points = (
                particle_object._cloth_prim_view.get_world_positions()
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
        except Exception as e2:
            logger.error(f"Error in get_object_particle_position: {e2}")
            return
    positions = (transformed_mesh_points[index_list] * 100).tolist()
    return positions


@step_interval(interval=50)
def success_checker_fold(
    particle_object, index_list=[8077, 1711, 2578, 3942, 8738, 588]
):
    p = get_object_particle_position(particle_object, index_list)
    success = (
        calculate_distance(p[0], p[4]) <= 10
        and calculate_distance(p[2], p[3]) <= 16
        and calculate_distance(p[1], p[5]) <= 10
    )
    return bool(success)


def check_top_sleeve(p, success_distance):
    dist_0_4 = calculate_distance(p[0], p[4])
    dist_2_3 = calculate_distance(p[2], p[3])
    dist_1_5 = calculate_distance(p[1], p[5])
    dist_0_1 = calculate_distance(p[0], p[1])
    dist_4_5 = calculate_distance(p[4], p[5])
    cond1 = dist_0_4 <= success_distance[0]
    cond2 = dist_2_3 <= success_distance[1]
    cond3 = dist_1_5 <= success_distance[2]
    cond4 = dist_0_1 >= success_distance[3]
    cond5 = dist_4_5 >= success_distance[4]

    details = {
        "condition_1": {
            "description": f"dist(p[0], p[4]) = {dist_0_4:.2f} <= {success_distance[0]}",
            "value": dist_0_4,
            "threshold": success_distance[0],
            "passed": cond1,
        },
        "condition_2": {
            "description": f"dist(p[2], p[3]) = {dist_2_3:.2f} <= {success_distance[1]}",
            "value": dist_2_3,
            "threshold": success_distance[1],
            "passed": cond2,
        },
        "condition_3": {
            "description": f"dist(p[1], p[5]) = {dist_1_5:.2f} <= {success_distance[2]}",
            "value": dist_1_5,
            "threshold": success_distance[2],
            "passed": cond3,
        },
        "condition_4": {
            "description": f"dist(p[0], p[1]) = {dist_0_1:.2f} >= {success_distance[3]}",
            "value": dist_0_1,
            "threshold": success_distance[3],
            "passed": cond4,
        },
        "condition_5": {
            "description": f"dist(p[4], p[5]) = {dist_4_5:.2f} >= {success_distance[4]}",
            "value": dist_4_5,
            "threshold": success_distance[4],
            "passed": cond5,
        },
    }

    return cond1 and cond2 and cond3 and cond4 and cond5, details

def check_pant_long(p, success_distance):
    dist_0_4 = calculate_distance(p[0], p[4])
    dist_0_2 = calculate_distance(p[0], p[2])   
    dist_1_3 = calculate_distance(p[1], p[3])
    dist_1_5 = calculate_distance(p[1], p[5])
    cond1 = dist_0_4 <= success_distance[0]
    cond2 = dist_0_2 >= success_distance[1]
    cond3 = dist_1_3 >= success_distance[2]
    cond4 = dist_1_5 <= success_distance[3]
    details = {
        "condition_1": {
            "description": f"dist(p[0], p[4]) = {dist_0_4:.2f} <= {success_distance[0]}",
            "value": dist_0_4,
            "threshold": success_distance[0],
            "passed": cond1,
        },
        "condition_2": {
            "description": f"dist(p[0], p[2]) = {dist_0_2:.2f} >= {success_distance[1]}",
            "value": dist_0_2,
            "threshold": success_distance[1],
            "passed": cond2,
        },
        "condition_3": {
            "description": f"dist(p[1], p[3]) = {dist_1_3:.2f} >= {success_distance[2]}",
            "value": dist_1_3,
            "threshold": success_distance[2],
            "passed": cond3,
        },
        "condition_4": {
            "description": f"dist(p[1], p[5]) = {dist_1_5:.2f} <= {success_distance[3]}",
            "value": dist_1_5,
            "threshold": success_distance[3],
            "passed": cond4,
        },
    }
    return cond1 and cond2 and cond3 and cond4, details

def check_pant_short(p, success_distance):
    dist_0_1 = calculate_distance(p[0], p[1])
    dist_4_5 = calculate_distance(p[4], p[5])
    dist_0_4 = calculate_distance(p[0], p[4])
    dist_1_5 = calculate_distance(p[1], p[5])
    cond1 = dist_0_1 <= success_distance[0]
    cond2 = dist_4_5 <= success_distance[1]
    cond3 = dist_0_4 >= success_distance[2]
    cond4 = dist_1_5 >= success_distance[3]

    details = {
        "condition_1": {
            "description": f"dist(p[0], p[1]) = {dist_0_1:.2f} <= {success_distance[0]}",
            "value": dist_0_1,
            "threshold": success_distance[0],
            "passed": cond1,
        },
        "condition_2": {
            "description": f"dist(p[4], p[5]) = {dist_4_5:.2f} <= {success_distance[1]}",
            "value": dist_4_5,
            "threshold": success_distance[1],
            "passed": cond2,
        },
        "condition_3": {
            "description": f"dist(p[0], p[4]) = {dist_0_4:.2f} >= {success_distance[2]}",
            "value": dist_0_4,
            "threshold": success_distance[2],
            "passed": cond3,
        },
        "condition_4": {
            "description": f"dist(p[1], p[5]) = {dist_1_5:.2f} >= {success_distance[3]}",
            "value": dist_1_5,
            "threshold": success_distance[3],
            "passed": cond4,
        },
    }
    return cond1 and cond2 and cond3 and cond4, details

@step_interval(interval=50)
def success_checker_garment_fold(particle_object, garment_type: str):
    check_point_indices = particle_object.check_points  # list[int]
    raw_success_distance = particle_object.success_distance  # list[int]
    current_scale = float(particle_object.init_scale[0])
    success_distance = [d * current_scale for d in raw_success_distance]
    p = get_object_particle_position(particle_object, check_point_indices)

    if garment_type == "top-long-sleeve" or garment_type == "top-short-sleeve":
        success, details = check_top_sleeve(p, success_distance)
    elif garment_type == "short-pant":
        success, details = check_pant_short(p, success_distance)
    elif garment_type == "long-pant":
        success, details = check_pant_long(p, success_distance)
    else:
        raise ValueError(f"Unknown garment_type: {garment_type}")

    result = {
        "success": bool(success),
        "garment_type": garment_type,
        "thresholds": success_distance,
        "details": details,
    }

    return result


@step_interval(interval=50)
def success_checker_fling(
    particle_object, index_list=[8077, 1711, 2578, 3942, 8738, 588]
):
    p = get_object_particle_position(particle_object, index_list)

    def xy_distance(a, b):
        return np.linalg.norm(np.array(a[:2]) - np.array(b[:2]))

    def z_distance(a, b):
        return abs(a[2] - b[2])

    success = (
        xy_distance(p[0], p[4]) > 18
        and z_distance(p[0], p[4]) < 2
        and xy_distance(p[1], p[5]) > 18
        and z_distance(p[1], p[5]) < 2
    )

    return bool(success)


@step_interval(interval=30)
def success_checker_burger(beef_pos, plate_pos):
    diff_xy = beef_pos[:, :2] - plate_pos[:, :2]
    dist_xy = torch.linalg.norm(diff_xy, dim=-1)

    # z distance
    diff_z = torch.abs(beef_pos[:, 2] - plate_pos[:, 2])

    # Success condition: xy < 0.045 and z < 0.03
    success_mask = (dist_xy < 0.045) & (diff_z < 0.03)
    success = success_mask.any().item()

    return bool(success)


@step_interval(interval=6)
def success_checker_cut(sausage_count: int) -> bool:
    return sausage_count >= 2
