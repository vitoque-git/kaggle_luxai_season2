# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
import sys

import numpy as np
def pr(*args, sep=' ', end='\n', force=False):  # print conditionally
    if (True or force): # change first parameter to False to disable logging
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
    if direction== 1: return 3
    elif direction== 2: return 4
    elif direction== 3: return 1
    elif direction== 4: return 2
    else: return 0

def get_next_pos(pos, direction):
    move_deltas = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]
    new_loc = np.array(pos) + move_deltas[direction]
    return (new_loc[0],new_loc[1])

def is_day(turn):
    return turn % 50 <= 30

def if_is_day(turn, value_if_day, value_if_night):
    if is_day(turn):
        return value_if_day
    else:
        return value_if_night