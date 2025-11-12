import random

def generate_unique_coordinates(n, upper_bound, lower_bound = 0):
    cords = []

    for _ in range(n):
        while True:
            x = random.randint(lower_bound, upper_bound)
            y = random.randint(lower_bound, upper_bound)

            cord = [x, y]
            if cord in cords:
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
    