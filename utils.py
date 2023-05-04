import dgl
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import time
from dataclasses import dataclass

@dataclass
class RunConfig:
    rank: int = 0
    world_size: int = 1
    in_feats: int = -1
    hid_feats: int = 128
    num_classes: int = -1# output feature size
    batch_size: int = 1024
    total_epoch: int = 10
    save_every: int = 10
    fanouts: list[int] = None
    graph_name: str = "ogbn-arxiv"
    log_path: str = "log.csv" # logging output path
    checkpt_path: str = "checkpt.pt" # checkpt path
    
    
def get_size(tensor: torch.Tensor) -> int:
    shape = tensor.shape
    size = 1
    if torch.float32 == tensor.dtype or torch.int32 == tensor.dtype:
        size *= 4
    elif torch.float64 == tensor.dtype or torch.int64 == tensor.dtype:
        size *= 8
    for dim in shape:
        size *= dim
    return size

def get_size_str(tensor: torch.Tensor) -> str:
    size = get_size(tensor)
    if size < 1e3:
        return f"{size / 1000.0} KB"
    elif size < 1e6:
        return f"{size / 1000.0} KB"
    elif size < 1e9:
        return f"{size / 1000000.0} MB"
    else:
        return f"{size / 1000000000.0} GB"
    
# This function split the feature data horizontally
# each node's data is partitioned into 'world_size' chunks
# return the partition corresponding to the 'rank'
# Input args:
# rank: [0, world_size - 1]
def get_local_feat(rank: int, world_size:int, feat: torch.Tensor) -> torch.Tensor:
    assert(feat.shape[1] % world_size == 0)
    step = int(feat.shape[1] / world_size)
    start_idx = rank * step
    end_idx = start_idx + step
    return feat[:, start_idx : end_idx]


def get_train_dataloader(rank: int, world_size: int, \
                         sampler: dgl.dataloading.neighbor_sampler.NeighborSampler, \
                         graph: dgl.DGLGraph, \
                         train_nids: torch.Tensor) -> dgl.dataloading.dataloader.DataLoader:
    device = torch.device(f"cuda:{rank}")
    train_dataloader = dgl.dataloading.DataLoader(
        # The following arguments are specific to DGL's DataLoader.
        graph,              # The graph
        train_nids,         # The node IDs to iterate over in minibatches
        sampler,            # The neighbor sampler
        device=device,      # Put the sampled MFGs on CPU or GPU
        use_ddp=world_size > 1, # enable ddp if using mutiple gpus
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=1024,    # Batch size
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=False,    # Whether to drop the last incomplete batch
        num_workers=0       # Number of sampler processes
    )
    return train_dataloader