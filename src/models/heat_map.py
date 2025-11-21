import torch.nn.functional as F
import torch
import torch.optim as optim
import os
from envs.minesweeper import MinesweeperEnv
from utils.constants import DEVICE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import utils.helper as helper
import pickle


class BombHeatMap:
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
                prediction = self.net(min_array)
                prediction_percent = F.softmax(prediction)
                confidence_matrix[:, x, y] = prediction_percent
        return confidence_matrix

    def train(self, dataset_name, epochs: int = 5, mini_batch_size: int = 100):
        if os.path.exists(f"src/data/{dataset_name}.pt"):
            dataset = torch.load(f"src/data/{dataset_name}.pt")
            data = dataset["data"]
            labels = dataset["labels"]

            X_train, X_test, y_train, y_test = train_test_split(
                data, labels, test_size=0.20
            )

            data_batches = X_train.split(mini_batch_size, dim=0)
            label_batches = y_train.split(mini_batch_size, dim=0)
        else:
            raise FileNotFoundError(f"No dataset with name {dataset_name}.")
        for epoch in range(epochs):
            print(f"\repoch {epoch}/{epochs}", end="")
            for data_batch, label_batch in zip(data_batches, label_batches):
                self.optimizer.zero_grad()
                output = self.net(data_batch)
                output = F.softmax(output, dim=1)

                loss = F.cross_entropy(output, label_batch)
                loss.backward()

                self.optimizer.step()

        print()
        print("training finished")
        print("evaluate model")
        with torch.no_grad():
            y_pred = self.net(X_test)
            y_pred = y_pred.max(dim=1)[1]
            y_test = y_test.max(dim=1)[1]

            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            acc = accuracy_score(y_test, y_pred)
            print(acc)

    def load_dataset(self, dataset_name):
        if not os.path.exists(f"src/data/{dataset_name}.pt"):
            raise FileNotFoundError
        dataset = torch.load(f"src/data/{dataset_name}.pt")
        data = dataset["data"]
        labels = dataset["labels"]
        return data, labels

    def append_to_dataset(
        self,
        env: MinesweeperEnv,
        dataset_name=None,
        repeats: int = 10,
        kernel_size: int = 1,
        bombs_per_iter: int = 10,
        saves_per_iter: int = 10,
    ):
        if os.path.exists(f"src/data/{dataset_name}.pt"):
            dataset = torch.load(f"src/data/{dataset_name}.pt")
            data = dataset["data"]
            labels = dataset["labels"]
        else:
            data = torch.empty(0)
            labels = torch.empty(0)

        new_data = []
        new_labels = []

        for episode in range(repeats):
            env.reset()
            next_state, _, done, _, info = env.step(-1)
            state = next_state
            done = False
            stop = False
            while not done and not stop:
                state = torch.tensor(
                    state, dtype=torch.float32, device=DEVICE
                ).unsqueeze(0)
                temp = F.pad(
                    input=state,
                    pad=(
                        kernel_size,
                        kernel_size,
                        kernel_size,
                        kernel_size,
                        0,
                        0,
                        0,
                        0,
                    ),
                    value=-1,
                )

                master_board = info["master_board"]
                save_actions = info["save_actions"]
                bomb_actions = info["bomb_actions"]

                if len(save_actions[0]) > saves_per_iter:
                    save_sample = np.random.choice(save_actions[0], saves_per_iter)
                    bomb_sample = np.random.choice(bomb_actions[0], bombs_per_iter)
                else:
                    stop = True

                for action in save_sample:
                    x, y = helper.action_to_index(action, master_board.shape)
                    x, y = x + kernel_size, y + kernel_size
                    min_array = temp[
                        :,
                        :,
                        x - (kernel_size) : x + (kernel_size + 1),
                        y - (kernel_size) : y + (kernel_size + 1),
                    ]
                    new_data.append(min_array)
                    label = torch.tensor([0, 1]).unsqueeze(0)
                    new_labels.append(label)

                for action in bomb_sample:
                    x, y = helper.action_to_index(action, master_board.shape)
                    x, y = x + kernel_size, y + kernel_size
                    min_array = temp[
                        :,
                        :,
                        x - (kernel_size) : x + (kernel_size + 1),
                        y - (kernel_size) : y + (kernel_size + 1),
                    ]
                    new_data.append(min_array)
                    label = torch.tensor([1, 0]).unsqueeze(0)
                    new_labels.append(label)

                next_state, _, done, _, info = env.step(-1)
                state = torch.tensor(next_state, dtype=torch.float32, device=DEVICE)

            if episode % 5000 == 0:
                data = torch.cat([data] + new_data, dim=0)
                labels = torch.cat([labels] + new_labels, dim=0)
                new_data = []
                new_labels = []

            print(f"\rGame: {episode + 1}/{repeats}", end="")

        data = torch.cat([data] + new_data, dim=0)
        labels = torch.cat([labels] + new_labels, dim=0)

        torch.save({"data": data, "labels": labels}, f"src/data/{dataset_name}.pt")

        print()
        print(f"data shape: {data.shape}")
        print(f"labels shape: {labels.shape}")

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
