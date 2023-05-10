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
    global_in_feats: int = -1
    local_in_feats: int = -1
    hid_feats: int = 128
    num_classes: int = -1 # output feature size
    batch_size: int = 1024
    total_epoch: int = 30
    save_every: int = 30
    fanouts: list[int] = None
    graph_name: str = "ogbn-arxiv"
    log_path: str = "log.csv" # logging output path
    checkpt_path: str = "checkpt.pt" # checkpt path
    mode: int = 1 # runner version
    def set_logpath(self):
        self.log_path = f"{self.graph_name}_h{self.hid_feats}_b{self.batch_size}_v{self.mode}.csv"
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
    

def get_train_dataloader(config: RunConfig,
                         sampler: dgl.dataloading.neighbor_sampler.NeighborSampler, 
                         graph: dgl.DGLGraph, 
                         train_nids: torch.Tensor) -> dgl.dataloading.dataloader.DataLoader:
    device = torch.device(f"cuda:{config.rank}")
    # start_idx = config.rank * config.batch_size
    # step = config.world_size * config.batch_size
    # max_iter = int(train_nids.shape[0] / step)
    # cur_iter = int((train_nids.shape[0] - start_idx) / step) 
    # assert(cur_iter + 2 >= max_iter)
    # drop_last = max_iter == cur_iter
    drop_last = False
    train_dataloader = dgl.dataloading.DataLoader(
        # The following arguments are specific to DGL's DataLoader.
        graph,              # The graph
        train_nids,         # The node IDs to iterate over in minibatches
        sampler,            # The neighbor sampler
        device=device,      # Put the sampled MFGs on CPU or GPU
        use_ddp=config.world_size > 1, # enable ddp if using mutiple gpus
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=config.batch_size,    # Batch size
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=drop_last,    # Whether to drop the last incomplete batch
        num_workers=0       # Number of sampler processes
    )
    return train_dataloader

def get_valid_dataloader(config: RunConfig,
                         sampler: dgl.dataloading.neighbor_sampler.NeighborSampler, 
                         graph: dgl.DGLGraph, 
                         valid_nids: torch.Tensor,
                         use_ddp: bool=False) -> dgl.dataloading.dataloader.DataLoader:
    device = torch.device(f"cuda:{config.rank}")
    train_dataloader = dgl.dataloading.DataLoader(
        # The following arguments are specific to DGL's DataLoader.
        graph,              # The graph
        valid_nids,         # The node IDs to iterate over in minibatches
        sampler,            # The neighbor sampler
        device=device,      # Put the sampled MFGs on CPU or GPU
        use_ddp=use_ddp,
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=config.batch_size,    # Batch size
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=True,    # Whether to drop the last incomplete batch
        num_workers=0       # Number of sampler processes
    )
    return train_dataloader

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
    return feat[:, start_idx : end_idx].to(rank)
