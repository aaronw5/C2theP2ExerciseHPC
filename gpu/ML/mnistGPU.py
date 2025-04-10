import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    # Use NCCL backend for GPU communication.
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

# Define the same simple MLP for MNIST classification.
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image.
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def ddp_train(rank, world_size):
    # Set up device (each process uses a different GPU).
    device = torch.device(f"cuda:{rank}")
    print(f"Rank {rank}: Using device {device}")
    setup_ddp(rank, world_size)
    
    # Prepare dataset with transformation and DistributedSampler.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=64, sampler=sampler)
    
    # Create model, move it to the correct GPU, and wrap with DDP.
    model = SimpleMLP().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    num_epochs = 3  # Adjust as needed.
    
    # Only rank 0 times the training.
    if rank == 0:
        start_time = time.time()
    
    model.train()
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Ensure proper shuffling.
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to the GPU corresponding to this rank.
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Rank {rank}, Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    if rank == 0:
        elapsed_time = time.time() - start_time
        print(f"DDP GPU training time (rank 0): {elapsed_time:.2f} seconds")
    
    cleanup_ddp()

def main_ddp():
    world_size = 2  # Number of GPUs/processes.
    # Spawn multiple processes (one per GPU).
    mp.spawn(ddp_train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main_ddp()

