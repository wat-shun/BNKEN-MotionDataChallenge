import os
import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch  # root package
from torch.utils.data import Dataset, DataLoader  # dataset representation and loading
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast

device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
print(f"Using {device} device")

JOINT_NUM = 21
JOINT_DIM = 3
D_TYPE = 'float32'


def rotate_y(frame, rad):
    # 右手系における回転行列
    R = np.array([
        [np.cos(rad), 0, np.sin(rad)],
        [0, 1, 0],
        [-np.sin(rad), 0, np.cos(rad)]
    ])

    return np.dot(R, frame.T).T


class TrainDataSet(Dataset):
    def __init__(self, frames: int, is_actual=False, deg_step=5):
        self.df = pd.read_csv('input_data/train/train.csv', dtype={'motion': str})
        self.frames = frames
        parts = self.df.keys()[2:]

        self.X = []
        self.y = []

        rads = []
        if is_actual:
            rads.append(0)
        else:
            # for deg in range(deg_step, 360, deg_step):
            for deg in range(0, 360, deg_step):
                rads.append(np.deg2rad(deg))

        for _, v in tqdm(self.df.groupby('motion')):
            t = v[parts].values

            for r in rads:
                X = self.__make_blank_vec()

                for j in range(0, t.shape[0] - frames, frames):
                    X[0] = t[j].reshape(JOINT_NUM, JOINT_DIM)
                    X[frames] = t[j + frames].reshape(JOINT_NUM, JOINT_DIM)
                    if not is_actual:
                        X[0] = rotate_y(X[0], rad=r)
                        X[frames] = rotate_y(X[frames], rad=r)
                    self.X.append(X.astype(D_TYPE))

                    y = self.__make_blank_vec()
                    for p in range(0, self.frames + 1):
                        y[p] = t[j + p].reshape(JOINT_NUM, JOINT_DIM)
                        if not is_actual:
                            y[p] = rotate_y(y[p], rad=r)

                    self.y.append(y.astype(D_TYPE))

    def __make_blank_vec(self):
        return np.zeros((self.frames + 1, JOINT_NUM, JOINT_DIM))

    def __len__(self):
        assert len(self.X) == len(self.y)
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train(dataloader, model, loss_function, optimizer, scaler):
    size = len(dataloader.dataset)
    model.train()
    for i, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Backpropagation
        optimizer.zero_grad()

        with autocast():
            # Compute prediction error
            pred = model(X)
            loss = loss_function(pred, y)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 100.0)

        scaler.step(optimizer)
        scaler.update()

        # loss.backward()
        # optimizer.step()

        if (i+1) % 50 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f} [{(i+1) * len(X):>6d}/{size:>6d}] {datetime.datetime.now().strftime('%H:%M:%S')}")


def test(dataloader, model, loss_function):
    model.eval()
    test_loss = 0
    cnt = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_function(pred, y).item()
            cnt += len(X)
    test_loss /= cnt
    print(f"Avg loss: {test_loss:>8f}")
    print()


def iterate(train_dataloader, test_dataloader, model, loss_function, optimizer, epochs: int, target_dir: str):
    try:
        os.mkdir(f'models/{target_dir}')
    except Exception as e:
        print(e)

    try:
        model.load_state_dict(torch.load(f"models/{target_dir}/latest.pth"))
    except Exception as e:
        print(e)

    scaler = GradScaler()
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_function, optimizer, scaler)
        test(test_dataloader, model, loss_function)

        torch.save(model.state_dict(), f"models/{target_dir}/latest.pth")
        torch.save(model.state_dict(), f"models/{target_dir}/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pth")
        print("Saved PyTorch Model State to model.pth")
    print("Done!")
