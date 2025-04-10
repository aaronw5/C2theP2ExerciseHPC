import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def ddp_train(rank, world_size):
    # each process on its own GPU
    device = torch.device(f"cuda:{rank}")
    print(f"Rank {rank}: Using device {device}")
    setup_ddp(rank, world_size)

    # 1) Load the processed tensors
    processed_dir = os.path.join('MNIST', 'processed')
    train_tensor, train_targets = torch.load(os.path.join(processed_dir, 'training.pt'))
    # train_tensor: UInt8Tensor [N,28,28], train_targets: LongTensor [N]

    # 2) Preprocess: add channel dim, to float, scale & normalize
    train_tensor = train_tensor.unsqueeze(1).float().div(255.)
    mean, std = 0.1307, 0.3081
    train_tensor = (train_tensor - mean) / std

    # 3) Wrap in a TensorDataset
    train_ds = TensorDataset(train_tensor, train_targets)

    # 4) Create DistributedSampler + DataLoader
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler)

    # 5) Model, DDP wrapper, loss & optimizer
    model = SimpleMLP().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    num_epochs = 3

    # timing only on rank 0
    if rank == 0:
        start_time = time.time()

    model.train()
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Rank {rank}, Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    if rank == 0:
        elapsed = time.time() - start_time
        print(f"DDP GPU training time (rank 0): {elapsed:.2f} seconds")

    cleanup_ddp()

def main_ddp():
    world_size = 2
    mp.spawn(ddp_train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main_ddp()
