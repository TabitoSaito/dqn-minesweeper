import random
import hashlib
import os
import glob
import torch
from utils.constants import DEVICE
from webdataset.compat import WebDataset
from torch.utils.data import DataLoader
from webdataset.autodecode import basichandlers


def generate_unique_coordinates(
    n, upper_bound_x, upper_bound_y, lower_bound_x=0, lower_bound_y=0, except_=[]
):
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


def action_to_index(action, shape: tuple):
    row = int(action / shape[0])
    col = action % shape[1]
    return (row, col)

def split_from_key(key: str, train_pct=0.8):
    # deterministic hash in range [0,100)
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 100
    if h < int(train_pct * 100):
        return "train"
    else:
        return "test"
    
def find_next_shard_id(pattern):
    directory = os.path.dirname(pattern)
    basename = os.path.basename(pattern)

    glob_pattern = os.path.join(directory, basename.replace("%05d", "*"))
    files = glob.glob(glob_pattern)

    if not files:
        return 0

    nums = []
    for f in files:
        name = os.path.basename(f)
        num_str = name.split("-")[-1].split(".")[0]  # "00010" -> "00010"
        nums.append(int(num_str))

    return max(nums) + 1

def np_to_tensor_kernel(sample):
    arr, label = sample
    x = torch.from_numpy(arr).float().to(DEVICE)
    y = int(label) if isinstance(label, (bytes, str, int)) else int(label.item())
    return x, y

def np_to_tensor_board(sample):
    arr, label = sample
    x = torch.from_numpy(arr).float().to(DEVICE)
    y = torch.from_numpy(label).float().to(DEVICE)
    return x, y

def collate_fn_kernel(batch):
    xs = torch.stack([b[0] for b in batch], dim=0).to(DEVICE)
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long).to(DEVICE)
    return xs, ys

def collate_fn_board(batch):
    xs = torch.stack([b[0] for b in batch], dim=0).to(DEVICE)
    ys = torch.stack([b[1] for b in batch], dim=0).to(DEVICE)
    return xs, ys

def make_loader_kernel(pattern, batch_size=64, num_workers=4, shuffle_buffer=1000, shuffle=True):
    files = sorted(glob.glob(pattern))
    ds = WebDataset(files, empty_check=False, shardshuffle=False)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    ds = ds.decode(basichandlers)          # decode .npy into numpy arrays
    ds = ds.to_tuple("npy", "cls")   # get two fields
    ds = ds.map(np_to_tensor_kernel)        # convert numpy -> torch
    # Optionally use .batched() to speed up if decode is expensive then convert batch -> tensors
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn_kernel, num_workers=num_workers)

def make_loader_board(pattern, batch_size=64, num_workers=4, shuffle_buffer=1000, shuffle=True):
    files = sorted(glob.glob(pattern))
    ds = WebDataset(files, empty_check=False, shardshuffle=False)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    ds = ds.decode(basichandlers)          # decode .npy into numpy arrays
    ds = ds.to_tuple("a1.npy", "a2.npy")   # get two fields
    ds = ds.map(np_to_tensor_board)        # convert numpy -> torch
    # Optionally use .batched() to speed up if decode is expensive then convert batch -> tensors
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn_board, num_workers=num_workers)

