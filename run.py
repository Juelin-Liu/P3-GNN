# Train the model with mutiple gpus
# Dataset (topology + feature) is duplicated across all the GPUs
# Model is duplicated across all the GPUs
# Use Pytorch DDP to synchronize model parameters / gradients across GPUs
# Use NCCL as the backend for communication

import dgl
import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
from models.sage import SAGE
from trainer import Trainer
from utils import RunConfig, get_size_str
from torch.distributed import init_process_group, destroy_process_group
import os
import torch.multiprocessing as mp

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def get_train_dataloader(config: RunConfig,
                         sampler: dgl.dataloading.neighbor_sampler.NeighborSampler, 
                         graph: dgl.DGLGraph, 
                         train_nids: torch.Tensor) -> dgl.dataloading.dataloader.DataLoader:
    device = torch.device(f"cuda:{config.rank}")
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
        drop_last=False,    # Whether to drop the last incomplete batch
        num_workers=0       # Number of sampler processes
    )
    return train_dataloader

def get_valid_dataloader(config: RunConfig,
                         sampler: dgl.dataloading.neighbor_sampler.NeighborSampler, 
                         graph: dgl.DGLGraph, 
                         valid_nids: torch.Tensor) -> dgl.dataloading.dataloader.DataLoader:
    device = torch.device(f"cuda:{config.rank}")
    train_dataloader = dgl.dataloading.DataLoader(
        # The following arguments are specific to DGL's DataLoader.
        graph,              # The graph
        valid_nids,         # The node IDs to iterate over in minibatches
        sampler,            # The neighbor sampler
        device=device,      # Put the sampled MFGs on CPU or GPU
        use_ddp=False,
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=config.batch_size,    # Batch size
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=False,    # Whether to drop the last incomplete batch
        num_workers=0       # Number of sampler processes
    )
    return train_dataloader

def main(rank:int, 
         world_size:int, 
         config: RunConfig,
         dataset: DglNodePropPredDataset):
    ddp_setup(rank, world_size)
    graph, node_labels = dataset[0]
    idx_split = dataset.get_idx_split()
    graph = graph.to(rank)
    node_labels = node_labels.to(rank)
    train_nids = idx_split['train'].to(rank)
    valid_nids = idx_split['valid'].to(rank)
    test_nids = idx_split['test'].to(rank)
    feat = graph.dstdata['feat'].to(rank)
    config.rank = rank
    config.world_size = world_size
    config.in_feats = feat.shape[1] # TODO: in P3 this should be feat.shape[0] / world_size
    config.num_classes = dataset.num_classes
    
    sampler = dgl.dataloading.NeighborSampler(config.fanouts)
    train_dataloader = get_train_dataloader(config, sampler, graph, train_nids)
    val_dataloader = get_valid_dataloader(config, sampler, graph, valid_nids)

    model = SAGE(in_feats=config.in_feats, hid_feats=config.hid_feats, num_layers=len(config.fanouts),out_feats=config.num_classes).to(config.rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(config, model, train_dataloader, val_dataloader, feat, node_labels, optimizer)
    trainer.train()
    destroy_process_group()
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=10, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=20, type=int, help='How often to save a snapshot')
    parser.add_argument('--hid_feats', default=64, type=int, help='Size of a hidden feature')
    parser.add_argument('--batch_size', default=1024, type=int, help='Input batch size on each device (default: 1024)')
    parser.add_argument('--graph_name', default="ogbn-arxiv", type=str, help="Input graph name any of['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']")
    args = parser.parse_args()
    config = RunConfig()

    world_size = torch.cuda.device_count()    
    config.batch_size = args.batch_size
    config.total_epoch = args.total_epochs
    config.hid_feats = args.hid_feats
    config.save_every = args.save_every
    config.graph_name = args.graph_name
    config.fanouts = [10, 10, 10]
    
    dataset = DglNodePropPredDataset(config.graph_name, root = "./dataset")
    mp.spawn(main, args=(world_size, config, dataset), nprocs=world_size)
    # main(rank, world_size, config, dataset)
    