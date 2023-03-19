# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
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

def is_day(turn):
    return turn % 50 <= 30