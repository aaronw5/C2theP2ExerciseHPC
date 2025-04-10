import os
import gzip
import struct
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import time

# --- same MNISTDataset as above ---
class MNISTDataset(Dataset):
    def __init__(self, root, train=True, download=True):
        self.root = root
        os.makedirs(root, exist_ok=True)
        if download:
            self._download()
        if train:
            img_f, lbl_f = 'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'
        else:
            img_f, lbl_f = 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'
        self.images = self._load_images(os.path.join(root, img_f))
        self.labels = self._load_labels(os.path.join(root, lbl_f))
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32)/255.0
        img = (img - 0.1307)/0.3081
        img = torch.from_numpy(img).unsqueeze(0)
        return img, int(self.labels[idx])
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
            _, num, rows, cols = struct.unpack('>IIII', f.read(16))
            buf = f.read(rows*cols*num)
            return np.frombuffer(buf, dtype=np.uint8).reshape(num, rows, cols)
    @staticmethod
    def _load_labels(path):
        with gzip.open(path, 'rb') as f:
            _, num = struct.unpack('>II', f.read(8))
            buf = f.read(num)
            return np.frombuffer(buf, dtype=np.uint8)

# --- same model ---
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

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def ddp_train(rank, world_size):
    device = torch.device(f'cuda:{rank}')
    print(f"Rank {rank} → {device}")
    setup_ddp(rank, world_size)

    ds = MNISTDataset(root='mnist_data', train=True, download=True)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=64, sampler=sampler)

    model = SimpleMLP().to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    num_epochs = 3

    if rank == 0:
        start = time.time()

    model.train()
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        loss_sum = 0.0
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print(f"Rank {rank} Epoch {epoch+1}/{num_epochs} Loss: {loss_sum/len(loader):.4f}")

    if rank == 0:
        elapsed = time.time() - start
        print(f"DDP training time (rank 0): {elapsed:.2f} sec")

    cleanup_ddp()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(ddp_train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
