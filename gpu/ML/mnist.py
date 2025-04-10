import os
import gzip
import struct
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time

# --- custom MNIST dataset (no torchvision) ---
class MNISTDataset(Dataset):
    def __init__(self, root, train=True, download=True):
        self.root = root
        os.makedirs(root, exist_ok=True)
        if download:
            self._download()
        # choose files
        if train:
            img_f, lbl_f = 'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'
        else:
            img_f, lbl_f = 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'
        self.images = self._load_images(os.path.join(root, img_f))
        self.labels = self._load_labels(os.path.join(root, lbl_f))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        # normalize like torchvision MNIST
        img = (img - 0.1307) / 0.3081
        img = torch.from_numpy(img).unsqueeze(0)  # [1,28,28]
        label = int(self.labels[idx])
        return img, label

    def _download(self):
        base = 'http://yann.lecun.com/exdb/mnist/'
        files = [
            'train-images-idx3-ubyte.gz',
            'train-labels-idx1-ubyte.gz',
            't10k-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte.gz'
        ]
        for fname in files:
            path = os.path.join(self.root, fname)
            if not os.path.exists(path):
                print(f"Downloading {fname}...")
                urllib.request.urlretrieve(base + fname, path)

    @staticmethod
    def _load_images(path):
        with gzip.open(path, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            buf = f.read(rows * cols * num)
            data = np.frombuffer(buf, dtype=np.uint8).reshape(num, rows, cols)
        return data

    @staticmethod
    def _load_labels(path):
        with gzip.open(path, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            buf = f.read(num)
            labels = np.frombuffer(buf, dtype=np.uint8)
        return labels

# --- simple MLP model ---
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def train_single_gpu():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # dataset + loader
    ds = MNISTDataset(root='mnist_data', train=True, download=True)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    model = SimpleMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    num_epochs = 3

    start_time = time.time()
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(loader):.4f}")
    total = time.time() - start_time
    print(f"Single‑GPU training time: {total:.2f} sec")

if __name__ == '__main__':
    train_single_gpu()
