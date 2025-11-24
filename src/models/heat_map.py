import torch.nn.functional as F
import torch
import torch.optim as optim
import os
from envs.minesweeper import MinesweeperEnv
from utils.constants import DEVICE, Identifier
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import utils.helper as helper
import pickle
from webdataset.writer import ShardWriter
import uuid
from tqdm import tqdm


class BombHeatMapKernel:
    def __init__(self, network) -> None:
        self.net = network
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

    def predict(self, data, kernel_size: int = 1):
        confidence_matrix = torch.zeros(2, data.shape[2], data.shape[3])
        temp = F.pad(
            input=data,
            pad=(kernel_size, kernel_size, kernel_size, kernel_size, 0, 0, 0, 0),
            value=-1,
        )
        for x in range(data.shape[2]):
            for y in range(data.shape[3]):
                x_pad, y_pad = x + kernel_size, y + kernel_size
                min_array = temp[
                    :,
                    :,
                    x_pad - (kernel_size) : x_pad + (kernel_size + 1),
                    y_pad - (kernel_size) : y_pad + (kernel_size + 1),
                ]
                with torch.no_grad():
                    prediction = self.net(min_array)
                    prediction_percent = F.softmax(prediction, dim=1)
                    confidence_matrix[:, x, y] = prediction_percent
        return confidence_matrix

    def train(self, dataset_name, epochs: int = 5, mini_batch_size: int = 128):
        train_loader = helper.make_loader_kernel(
            f"src/data/{dataset_name}/train/" + "shard-*.tar",
            batch_size=mini_batch_size,
            shuffle=True,
        )

        for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
            for data_batch, label_batch in train_loader:
                self.optimizer.zero_grad()
                output = self.net(data_batch)
                output = F.softmax(output, dim=1)
                output = output[:, 1]

                label_batch = label_batch.type(torch.float32)

                loss = F.cross_entropy(output, label_batch)
                loss.backward()

                self.optimizer.step()

        print("training finished")

    def evaluate(self, dataset_name, mini_batch_size=128):
        print("evaluate model")

        test_loader = helper.make_loader_kernel(
            f"src/data/{dataset_name}/test/" + "shard-*.tar",
            batch_size=mini_batch_size,
            shuffle=False,
        )

        with torch.no_grad():
            y_preds = []
            y_tests = []
            for data_batch, label_batch in test_loader:
                output = self.net(data_batch)
                y_pred = output.max(dim=1)[1]
                y_test = label_batch

                y_preds.append(y_pred)
                y_tests.append(y_test)

            y_pred = torch.cat(y_preds)
            y_test = torch.cat(y_tests)

            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            acc = accuracy_score(y_test, y_pred)
            print(acc)

    # TODO make YAML file for dataset to save config
    # TODO add num arrays in dataset counter
    def append_to_dataset(
        self,
        env: MinesweeperEnv,
        dataset_name=None,
        repeats: int = 10,
        kernel_size: int = 1,
        max_bombs_per_iter: int = 10,
        max_saves_per_iter: int = 10,
    ):
        os.makedirs(f"src/data/{dataset_name}/train/", exist_ok=True)
        os.makedirs(f"src/data/{dataset_name}/test/", exist_ok=True)

        train_pattern = f"src/data/{dataset_name}/train/" + "shard-%05d.tar"
        test_pattern = f"src/data/{dataset_name}/test/" + "shard-%05d.tar"

        train_shard = helper.find_next_shard_id(train_pattern)
        test_shard = helper.find_next_shard_id(test_pattern)

        train_writer = ShardWriter(
            f"src/data/{dataset_name}/train/" + "shard-%05d.tar",
            maxcount=100000,
            start_shard=train_shard,
            verbose=0
        )
        test_writer = ShardWriter(
            f"src/data/{dataset_name}/test/" + "shard-%05d.tar",
            maxcount=100000,
            start_shard=test_shard,
            verbose=0
        )

        for episode in tqdm(range(repeats), desc="Appending to dataset", unit="games"):
            env.reset()
            next_state, _, done, _, info = env.step(-1)
            state = next_state
            done = False
            saves_per_iter = max_saves_per_iter
            bombs_per_iter = max_bombs_per_iter
            while not done:
                temp = np.pad(
                    state,
                    pad_width=[
                        (0, 0),
                        (kernel_size, kernel_size),
                        (kernel_size, kernel_size),
                    ],
                    mode="constant",
                    constant_values=-1,
                )

                master_board = info["master_board"]
                save_actions = info["save_actions"]
                bomb_actions = info["bomb_actions"]

                if len(save_actions[0]) < saves_per_iter:
                    saves_per_iter = len(save_actions[0])
                    bombs_per_iter = len(save_actions[0])

                if saves_per_iter <= 0:
                    break

                save_sample = np.random.choice(save_actions[0], saves_per_iter)
                bomb_sample = np.random.choice(bomb_actions[0], bombs_per_iter)

                for action in save_sample:
                    x, y = helper.action_to_index(action, master_board.shape)
                    x, y = x + kernel_size, y + kernel_size
                    min_array = temp[
                        :,
                        x - (kernel_size) : x + (kernel_size + 1),
                        y - (kernel_size) : y + (kernel_size + 1),
                    ]
                    label = 0

                    key = f"s-{uuid.uuid4().hex}"
                    split = helper.split_from_key(key)
                    sink = {"train": train_writer, "test": test_writer}[split]
                    sink.write(
                        {"__key__": key, "npy": min_array, "cls": str(label).encode()}
                    )

                for action in bomb_sample:
                    x, y = helper.action_to_index(action, master_board.shape)
                    x, y = x + kernel_size, y + kernel_size
                    min_array = temp[
                        :,
                        x - (kernel_size) : x + (kernel_size + 1),
                        y - (kernel_size) : y + (kernel_size + 1),
                    ]
                    label = 1

                    key = f"b-{uuid.uuid4().hex}"
                    split = helper.split_from_key(key)
                    sink = {"train": train_writer, "test": test_writer}[split]
                    sink.write(
                        {"__key__": key, "npy": min_array, "cls": str(label).encode()}
                    )

                next_state, _, done, _, info = env.step(-1)
                state = next_state
        print("appending finished")

    def save(
        self,
        name=None,
    ):
        file_name = name if name else "checkpoint"
        temp = f"src/checkpoint/heat_map/{file_name}.tmp"
        checkpoint = {
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        with open(temp, "wb") as f:
            pickle.dump((checkpoint), f)
        os.replace(temp, f"src/checkpoint/heat_map/{file_name}.pkl")

    def load(self, name=None):
        file_name = name if name else "checkpoint"
        path = f"src/checkpoint/heat_map/{file_name}.pkl"
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        self.net.load_state_dict(checkpoint["model_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)

class BombHeatMapBoard:
    def __init__(self, network) -> None:
        self.net = network
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

    def predict(self, data, **kwarg):
        confidence_matrix = torch.zeros(2, data.shape[2], data.shape[3])
        prediction = self.net(data)
        prediction = F.softmax(prediction, dim=1)
        prediction = torch.reshape(prediction, confidence_matrix.shape[1:])
        confidence_matrix[0] = prediction
        confidence_matrix[1] = 1 - prediction
        
        return confidence_matrix

    def train(self, dataset_name, epochs: int = 5, mini_batch_size: int = 128):
        train_loader = helper.make_loader_board(
            f"src/data/{dataset_name}/train/" + "shard-*.tar",
            batch_size=mini_batch_size,
            shuffle=True,
        )

        for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
            for data_batch, label_batch in train_loader:
                self.optimizer.zero_grad()
                output = self.net(data_batch)
                output = F.softmax(output, dim=1)

                loss = F.cross_entropy(output, label_batch)
                loss.backward()

                self.optimizer.step()

        print("training finished")

    def evaluate(self, dataset_name, mini_batch_size=128):
        print("evaluate model")

        test_loader = helper.make_loader_board(
            f"src/data/{dataset_name}/test/" + "shard-*.tar",
            batch_size=mini_batch_size,
            shuffle=False,
        )

        with torch.no_grad():
            y_preds = []
            y_tests = []
            for data_batch, label_batch in test_loader:
                output = self.net(data_batch)
                y_pred = output.max(dim=1)[1]
                y_test = label_batch

                y_preds.append(y_pred)
                y_tests.append(y_test)

            y_pred = torch.cat(y_preds)
            y_test = torch.cat(y_tests)

            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            acc = accuracy_score(y_test, y_pred)
            print(acc)

    # TODO make YAML file for dataset to save config
    # TODO add num arrays in dataset counter
    def append_to_dataset(
        self,
        env: MinesweeperEnv,
        dataset_name=None,
        repeats: int = 10,
        kernel_size: int = 1,
        max_bombs_per_iter: int = 10,
        max_saves_per_iter: int = 10,
    ):
        os.makedirs(f"src/data/{dataset_name}/train/", exist_ok=True)
        os.makedirs(f"src/data/{dataset_name}/test/", exist_ok=True)

        train_pattern = f"src/data/{dataset_name}/train/" + "shard-%05d.tar"
        test_pattern = f"src/data/{dataset_name}/test/" + "shard-%05d.tar"

        train_shard = helper.find_next_shard_id(train_pattern)
        test_shard = helper.find_next_shard_id(test_pattern)

        train_writer = ShardWriter(
            f"src/data/{dataset_name}/train/" + "shard-%05d.tar",
            maxcount=100000,
            start_shard=train_shard,
            verbose=0
        )
        test_writer = ShardWriter(
            f"src/data/{dataset_name}/test/" + "shard-%05d.tar",
            maxcount=100000,
            start_shard=test_shard,
            verbose=0
        )

        for episode in tqdm(range(repeats), desc="Appending to dataset", unit="games"):
            env.reset()
            next_state, _, done, _, info = env.step(-1)
            state = next_state
            done = False
            while not done:

                master_board = info["master_board"]
                temp = np.zeros(shape=master_board.shape)

                temp[master_board == Identifier.BOMB.value] = 1

                key = f"s-{uuid.uuid4().hex}"
                split = helper.split_from_key(key)
                sink = {"train": train_writer, "test": test_writer}[split]
                sink.write(
                    {"__key__": key, "a1.npy": state, "a2.npy": temp.flatten()}
                )

                next_state, _, done, _, info = env.step(-1)
                state = next_state
        print("appending finished")

    def save(
        self,
        name=None,
    ):
        file_name = name if name else "checkpoint"
        temp = f"src/checkpoint/heat_map/{file_name}.tmp"
        checkpoint = {
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        with open(temp, "wb") as f:
            pickle.dump((checkpoint), f)
        os.replace(temp, f"src/checkpoint/heat_map/{file_name}.pkl")

    def load(self, name=None):
        file_name = name if name else "checkpoint"
        path = f"src/checkpoint/heat_map/{file_name}.pkl"
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        self.net.load_state_dict(checkpoint["model_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)
