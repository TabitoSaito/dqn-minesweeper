import random

def generate_unique_coordinates(n, upper_bound_x, upper_bound_y, lower_bound_x = 0, lower_bound_y = 0, except_ = []):
    cords = []

    for _ in range(n):
        while True:
            x = random.randint(lower_bound_x, upper_bound_x)
            y = random.randint(lower_bound_y, upper_bound_y)

            cord = [x, y]
            if cord in cords:
                continue
            elif cord[0] == except_[0] and cord[1] == except_[1]:
                continue
            else:
                cords.append(cord)
                break
    return list(map(list, zip(*cords)))

def index_in_bound(index: tuple[int, int], bound: tuple[int, int]):
    if not 0 <= index[0] < bound[0]:
        return False
    if not 0 <= index[1] < bound[1]:
        return False
    return True
    