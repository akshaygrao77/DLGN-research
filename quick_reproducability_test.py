from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
import random


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_id + worker_seed)
    random.seed(worker_id - worker_seed)


gen = torch.Generator()
gen.manual_seed(20)

gen2 = torch.Generator()
gen2.manual_seed(20)

dataset = TensorDataset(torch.arange(0, 100))
loader1 = DataLoader(dataset, batch_size=8, shuffle=True,
                     num_workers=2, generator=gen, worker_init_fn=seed_worker)
loader2 = DataLoader(dataset, batch_size=8, shuffle=True,
                     num_workers=2, generator=gen2, worker_init_fn=seed_worker)

print("l1", next(iter(loader1)))
print("l2", next(iter(loader2)))

print("l1", next(iter(loader1)))
print("l1", next(iter(loader1)))
print("l1", next(iter(loader1)))
print("l2", next(iter(loader2)))
