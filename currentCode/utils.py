# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
import sys

import numpy as np


def pr(*args, sep=' ', end='\n', force=False):  # print conditionally
    if (True or force):  # change first parameter to False to disable logging
        print(*args, sep=sep, file=sys.stderr)


def prx(*args): pr(*args, force=True)


def direction_to(src, target):
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) > abs(dy):
        if dx > 0:
            return 2
        else:
            return 4
    else:
        if dy > 0:
            return 3
        else:
            return 1


# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
def opposite_direction(direction):
    if direction == 1:
        return 3
    elif direction == 2:
        return 4
    elif direction == 3:
        return 1
    elif direction == 4:
        return 2
    else:
        return 0


def get_next_pos(pos, direction):
    move_deltas = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]
    new_loc = np.array(pos) + move_deltas[direction]
    return (new_loc[0], new_loc[1])


def is_day(turn):
    return turn % 50 <= 30


def if_is_day(turn, value_if_day, value_if_night):
    if is_day(turn):
        return value_if_day
    else:
        return value_if_night


# Manhattan Distance between one points and one vector, return a vector
def get_distance_vector(pos, points):
    return 2 * np.mean(np.abs(points - pos), 1)


# Manhattan Distance between two points
def get_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_map_distances(locations, pos):
    distances = get_distance_vector(pos, locations)
    sorted_loc = [locations[k] for k in np.argsort(distances)]
    closest_loc = sorted_loc[0]
    return closest_loc, sorted_loc


def get_straight_direction(unit, closest_tile):
    closest_tile = np.array(closest_tile)
    direction = direction_to(np.array(unit.pos), closest_tile)
    move_to = get_next_pos(unit.pos_location(), direction)
    return direction, move_to


def expand_point(opp_factories_areas, pos):
    x = pos[0]
    y = pos[1]
    opp_factories_areas.append((x - 1, y - 1))
    opp_factories_areas.append((x - 1, y))
    opp_factories_areas.append((x - 1, y + 1))
    opp_factories_areas.append((x, y - 1))
    opp_factories_areas.append((x, y))
    opp_factories_areas.append((x, y + 1))
    opp_factories_areas.append((x + 1, y - 1))
    opp_factories_areas.append((x + 1, y))
    opp_factories_areas.append((x + 1, y + 1))
