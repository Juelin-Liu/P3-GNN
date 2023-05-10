# Train the model with mutiple gpus
# Topology graph is duplicated across all the GPUs
# Feature data is horizontally partitioned across all the GPUs
# Each GPU has a feature size with: [#Nodes, Origin Feature Size / Total GPUs]
# Before every minibatch, feature is fetched from other GPUs for aggregation
# Model is duplicated across all the GPUs
# Use Pytorch DDP to synchronize model parameters / gradients across GPUs
# Use NCCL as the backend for communication

import dgl
import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
from models.sage import SAGE
from models.p3_sage import create_p3model, P3_SAGE
from trainer import Trainer
from p3_trainer import P3Trainer
from utils import *
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

def main_v1(rank:int, 
         world_size:int, 
         config: RunConfig,
         global_feat: torch.nn.Module, # CPU feature
         graph, node_labels, dataset):
    ddp_setup(rank, world_size)
    graph = graph.to(rank)
    idx_split = dataset.get_idx_split()
    node_labels = node_labels.to(rank)
    train_nids = idx_split['train'].to(rank).to(graph.idtype)
    valid_nids = idx_split['valid'].to(rank).to(graph.idtype)
    test_nids = idx_split['test'].to(rank).to(graph.idtype)
    feat = global_feat.to(rank)
    print(f"Rank: {rank} | Local Feature Size: {get_size_str(feat)}")

    config.rank = rank
    config.mode = 1
    config.world_size = world_size
    config.global_in_feats = int(feat.shape[1]) # TODO: in P3 this should be feat.shape[0] / world_size
    config.num_classes = dataset.num_classes
    config.set_logpath()
    sampler = dgl.dataloading.NeighborSampler(config.fanouts)
    train_dataloader = get_train_dataloader(config, sampler, graph, train_nids)
    val_dataloader = get_valid_dataloader(config, sampler, graph, valid_nids)

    model = SAGE(in_feats=config.global_in_feats, hid_feats=config.hid_feats, num_layers=len(config.fanouts),out_feats=config.num_classes).to(config.rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(config, model, train_dataloader, val_dataloader, feat, global_feat, node_labels, optimizer, graph.idtype)
    trainer.train_v1()
    destroy_process_group()
    
def main_v2(rank:int, 
         world_size:int, 
         config: RunConfig,
         global_feat: torch.Tensor, # CPU feature
         graph, node_labels, dataset):
    ddp_setup(rank, world_size)
    graph = graph.to(rank)
    idx_split = dataset.get_idx_split()
    node_labels = node_labels.to(rank)
    train_nids = idx_split['train'].to(rank).to(graph.idtype)
    valid_nids = idx_split['valid'].to(rank).to(graph.idtype)
    test_nids = idx_split['test'].to(rank).to(graph.idtype)
    local_feat = get_local_feat(rank, world_size, global_feat)
    print(f"Rank: {rank} | Local Feature Size: {get_size_str(local_feat)}")

    config.rank = rank
    config.world_size = world_size
    config.global_in_feats = global_feat.shape[1]
    config.local_in_feats = local_feat.shape[1]
    config.num_classes = dataset.num_classes
    config.mode = 2
    config.set_logpath()
    sampler = dgl.dataloading.NeighborSampler(config.fanouts)
    train_dataloader = get_train_dataloader(config, sampler, graph, train_nids)
    val_dataloader = get_valid_dataloader(config, sampler, graph, valid_nids)

    model = SAGE(in_feats=config.global_in_feats, hid_feats=config.hid_feats, num_layers=len(config.fanouts),out_feats=config.num_classes).to(config.rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(config, model, train_dataloader, val_dataloader, local_feat, global_feat, node_labels, optimizer, graph.idtype)
    trainer.train_v2()
    destroy_process_group()

def main_v3(rank:int, 
         world_size:int, 
         config: RunConfig,
         global_feat: torch.Tensor, # CPU feature
         graph, node_labels, dataset):
    ddp_setup(rank, world_size)
    graph = graph.to(rank)
    idx_split = dataset.get_idx_split()
    node_labels = node_labels.to(rank)
    train_nids = idx_split['train'].to(rank).to(graph.idtype)
    valid_nids = idx_split['valid'].to(rank).to(graph.idtype)
    test_nids = idx_split['test'].to(rank).to(graph.idtype)
    local_feat = get_local_feat(rank, world_size, global_feat)
    print(f"Rank: {rank} | Local Feature Size: {get_size_str(local_feat)}")

    config.rank = rank
    config.world_size = world_size
    config.global_in_feats = global_feat.shape[1]
    config.local_in_feats = local_feat.shape[1]
    config.num_classes = dataset.num_classes
    config.mode = 3
    config.set_logpath()
    sampler = dgl.dataloading.NeighborSampler(config.fanouts)
    train_dataloader = get_train_dataloader(config, sampler, graph, train_nids)
    val_dataloader = get_valid_dataloader(config, sampler, graph, valid_nids, use_ddp=True)
    local_model, global_model = create_p3model(config.local_in_feats, hid_feats=config.hid_feats, num_layers=len(config.fanouts), num_classes=config.num_classes)
    global_optimizer = torch.optim.Adam(global_model.parameters(), lr=1e-3)
    local_optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-3)
    trainer = P3Trainer(config, global_model, local_model, train_dataloader, val_dataloader, local_feat, global_feat, node_labels, global_optimizer, local_optimizer, nid_dtype=torch.int32)
    trainer.train_v3()
    destroy_process_group()
      
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=101, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=150, type=int, help='How often to save a snapshot')
    parser.add_argument('--hid_feats', default=256, type=int, help='Size of a hidden feature')
    parser.add_argument('--batch_size', default=1024, type=int, help='Input batch size on each device (default: 1024)')
    parser.add_argument('--mode', default=3, type=int, help='Runner mode (1: full replicate; 2: p3 partition; 3: p3 compute + partition)')
    parser.add_argument('--graph_name', default="ogbn-arxiv", type=str, help="Input graph name any of['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']")
    args = parser.parse_args()
    config = RunConfig()

    world_size = torch.cuda.device_count()
    # world_size = 1
    print(f"using {world_size} GPUs")
    config.batch_size = args.batch_size
    config.total_epoch = args.total_epochs
    config.hid_feats = args.hid_feats
    config.save_every = args.save_every
    config.graph_name = args.graph_name
    config.fanouts = [20, 20, 20]
    print("loading data")
    dataset = DglNodePropPredDataset(config.graph_name, root="/data/juelin/project/P3-GNN/dataset")
    graph, node_labels = dataset[0]
    global_feat = graph.dstdata.pop("feat")
    graph = graph.int()
    print("Global Feature Size: ", get_size_str(global_feat))
    print("Graph IdType: ", graph.idtype)
    if args.mode == 1:
        mp.spawn(main_v1, args=(world_size, config, global_feat, graph, node_labels, dataset), nprocs=world_size, daemon=True)
    elif args.mode == 2:
        mp.spawn(main_v2, args=(world_size, config, global_feat, graph, node_labels, dataset), nprocs=world_size, daemon=True)
    elif args.mode == 3:
        mp.spawn(main_v3, args=(world_size, config, global_feat, graph, node_labels, dataset), nprocs=world_size, daemon=True)