# Train the model with mutiple gpus
# Topology graph is duplicated across all the GPUs
# Feature data is horizontally partitioned across all the GPUs
# Each GPU has a feature size with: [#Nodes, Origin Feature Size / Total GPUs]
# Before every minibatch, feature is fetched from other GPUs for aggregation
# Model is duplicated across all the GPUs
# Use Pytorch DDP to synchronize model parameters / gradients across GPUs
# Use NCCL as the backend for communication
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
import dgl
import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
from models.sage import *
from dgl_trainer import DglTrainer
from p2_trainer import P2Trainer
from p3_trainer import P3Trainer
from quiver_trainer import QuiverTrainer
import quiver
import gc
from utils import *
from torch.distributed import init_process_group, destroy_process_group, barrier
import os
import torch.multiprocessing as mp
import math
from ogb.lsc import MAG240MDataset
from utils import QuiverDglSageSample

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
def main_v0(rank:int, 
         world_size:int, 
         config: RunConfig,
         global_feat: quiver.Feature, # CPU feature
         sampler: quiver.pyg.GraphSageSampler, 
         node_labels, idx_split):
    ddp_setup(rank, world_size)
    # graph = dgl.hetero_from_shared_memory("graph")
    # if config.topo == 'gpu':
    #     graph = graph.formats(["csc"])
    #     graph = graph.to(rank)
    node_labels = node_labels.to(rank)
    train_nids = idx_split['train'].to(torch.int64)
    valid_nids = idx_split['valid'].to(torch.int64)
    test_nids = idx_split['test'].to(torch.int64)
    train_nids = partition_ids(rank, world_size, train_nids)
    valid_nids = partition_ids(rank, world_size, valid_nids)
    # print(f"{rank=} {train_nids.shape=} {valid_nids.shape=}")
    config.rank = rank
    config.mode = 0
    config.world_size = world_size
    config.set_logpath()
    # config.global_in_feats = dataset. # TODO: in P3 this should be feat.shape[0] / world_sizeconfig.set_logpath()
    # sampler = dgl.dataloading.NeighborSampler(config.fanouts)
    # train_dataloader = get_train_dataloader(config, sampler, graph, train_nids, use_uva=config.uva_sample())
    # val_dataloader = get_valid_dataloader(config, sampler, graph, valid_nids, use_uva=config.uva_sample())
    train_dataloader = QuiverDglSageSample(rank=rank, batch_size=config.batch_size, nids=train_nids, sampler=sampler)
    val_dataloader = QuiverDglSageSample(rank=rank, batch_size=config.batch_size, nids=valid_nids, sampler=sampler)
    model = Sage(in_feats=config.global_in_feats, hid_feats=config.hid_feats, num_layers=len(config.fanouts),out_feats=config.num_classes).to(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = QuiverTrainer(config, model, train_dataloader, val_dataloader, global_feat, node_labels, optimizer, torch.int64)
    trainer.train()
    destroy_process_group()
    
def main_v1(rank:int, 
         world_size:int, 
         config: RunConfig,
         feat: torch.Tensor,
         sampler: quiver.pyg.GraphSageSampler, 
         node_labels: torch.Tensor, 
         idx_split):
    ddp_setup(rank, world_size)
    # graph = dgl.hetero_from_shared_memory("graph")
    # if config.topo == 'gpu':
    #     graph = graph.formats(["csc"])
    #     graph = graph.to(rank)
    
    node_labels = node_labels.to(rank)
    train_nids = idx_split['train'].to(torch.int64)
    valid_nids = idx_split['valid'].to(torch.int64)
    test_nids = idx_split['test'].to(torch.int64)
    train_nids = partition_ids(rank, world_size, train_nids)
    valid_nids = partition_ids(rank, world_size, valid_nids)
    # print(f"{rank=} {train_nids.shape=} {valid_nids.shape=}")

    config.rank = rank
    config.mode = 1
    config.world_size = world_size
    config.global_in_feats = int(feat.shape[1])
    
    config.set_logpath()
    # sampler = dgl.dataloading.NeighborSampler(config.fanouts)
    # train_dataloader = get_train_dataloader(config, sampler, graph, train_nids, use_uva=config.uva_sample())
    # val_dataloader = get_valid_dataloader(config, sampler, graph, valid_nids, use_uva=config.uva_sample())
    train_dataloader = QuiverDglSageSample(rank=rank, batch_size=config.batch_size, nids=train_nids, sampler=sampler)
    val_dataloader = QuiverDglSageSample(rank=rank, batch_size=config.batch_size, nids=valid_nids, sampler=sampler)
    model = Sage(in_feats=config.global_in_feats, hid_feats=config.hid_feats, num_layers=len(config.fanouts),out_feats=config.num_classes).to(config.rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = DglTrainer(config, model, train_dataloader, val_dataloader, feat, node_labels, optimizer, torch.int64)
    trainer.train()
    destroy_process_group()
    
    
def main_v2(rank:int, 
         world_size:int, 
         config: RunConfig,
         loc_feats: list[torch.Tensor], # CPU feature
         sampler: quiver.pyg.GraphSageSampler,
         node_labels, idx_split):
    ddp_setup(rank, world_size)
    # graph = dgl.hetero_from_shared_memory("graph")
    # if config.topo == 'gpu':
    #     graph = graph.formats(["csc"])
    #     graph = graph.to(rank)
    node_labels = node_labels.to(rank)
    train_nids = idx_split['train'].to(torch.int64)
    valid_nids = idx_split['valid'].to(torch.int64)
    test_nids = idx_split['test'].to(torch.int64)
    train_nids = partition_ids(rank, world_size, train_nids)
    valid_nids = partition_ids(rank, world_size, valid_nids)
    loc_feat = torch.load(os.path.join(dataset.dir, "processed", "paper", f"node_feat_w{world_size}_r{rank}.pt")).type(torch.float32)
    if config.uva_feat():
        loc_feat = loc_feat.pin_memory()
    else:
        loc_feat = loc_feat.to(rank)
        
    config.rank = rank
    config.world_size = world_size
    config.mode = 2
    
    config.set_logpath()
    # sampler = dgl.dataloading.NeighborSampler(config.fanouts)
    # train_dataloader = get_train_dataloader(config, sampler, graph, train_nids, use_uva=config.uva_sample())
    # val_dataloader = get_valid_dataloader(config, sampler, graph, valid_nids, use_uva=config.uva_sample())
    model = Sage(in_feats=config.global_in_feats, hid_feats=config.hid_feats, num_layers=len(config.fanouts),out_feats=config.num_classes).to(config.rank)
    train_dataloader = QuiverDglSageSample(rank=rank, batch_size=config.batch_size, nids=train_nids, sampler=sampler)
    val_dataloader = QuiverDglSageSample(rank=rank, batch_size=config.batch_size, nids=valid_nids, sampler=sampler)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = P2Trainer(config, model, train_dataloader, val_dataloader, loc_feat, node_labels, optimizer, torch.int64)
    trainer.train()
    destroy_process_group()

def main_v3(rank:int, 
         world_size:int, 
         config: RunConfig,
         loc_feats: list[torch.Tensor], # CPU feature
         sampler: quiver.pyg.GraphSageSampler, 
         node_labels, idx_split):
    ddp_setup(rank, world_size)
    # graph = dgl.hetero_from_shared_memory("graph")
    # if config.topo == 'gpu':
    #     graph = graph.formats(["csc"])
    #     graph = graph.to(rank)
        
    node_labels = node_labels.to(rank)
    train_nids = idx_split['train'].to(torch.int64)
    valid_nids = idx_split['valid'].to(torch.int64)
    test_nids = idx_split['test'].to(torch.int64)
    train_nids = partition_ids(rank, world_size, train_nids)
    valid_nids = partition_ids(rank, world_size, valid_nids)
    # test_nids = idx_split['test'].to(rank).to(torch.int64)
    loc_feat = torch.load(os.path.join(dataset.dir, "processed", "paper", f"node_feat_w{world_size}_r{rank}.pt")).type(torch.float32)
    if config.uva_feat():
        loc_feat = loc_feat.pin_memory()
    else:
        loc_feat = loc_feat.to(rank)
        
    config.rank = rank
    config.world_size = world_size
    config.mode = 3
    config.set_logpath()
    # sampler = dgl.dataloading.NeighborSampler(config.fanouts)
    # train_dataloader = get_train_dataloader(config, sampler, graph, train_nids, use_uva=config.uva_sample())
    # val_dataloader = get_valid_dataloader(config, sampler, graph, valid_nids, use_uva=config.uva_sample())
    local_model, global_model = create_sage_p3(rank, config.local_in_feats, hid_feats=config.hid_feats, num_layers=len(config.fanouts), num_classes=config.num_classes)
    global_optimizer = torch.optim.Adam(global_model.parameters(), lr=1e-3)
    local_optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-3)
    train_dataloader = QuiverDglSageSample(rank=rank, batch_size=config.batch_size, nids=train_nids, sampler=sampler)
    val_dataloader = QuiverDglSageSample(rank=rank, batch_size=config.batch_size, nids=valid_nids, sampler=sampler)
    trainer = P3Trainer(config, global_model, local_model, train_dataloader, val_dataloader, loc_feat, node_labels, global_optimizer, local_optimizer, nid_dtype=torch.int64)
    trainer.train()
    destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=6, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=150, type=int, help='How often to save a snapshot')
    parser.add_argument('--hid_feats', default=256, type=int, help='Size of a hidden feature')
    parser.add_argument('--batch_size', default=1024, type=int, help='Input batch size on each device (default: 1024)')
    parser.add_argument('--mode', default=0, type=int, help='Runner mode (0: Quiver Extract; 1: full replicate; 2: p3 partition; 3: p3 compute + partition)')
    parser.add_argument('--nprocs', default=8, type=int, help='Number of GPUs / processes')
    parser.add_argument('--topo', default="UVA", type=str, help='UVA, GPU, CPU', choices=["CPU", "UVA", "GPU"])
    parser.add_argument('--feat', default="UVA", type=str, help='UVA, GPU, CPU', choices=["CPU", "UVA", "GPU"])
    args = parser.parse_args()
    config = RunConfig()
    world_size = min(args.nprocs, torch.cuda.device_count())
    print(f"using {world_size} GPUs")
    print("start loading data")
    
    load_start = time.time()
    dataset = MAG240MDataset(root="/home/ubuntu/dataset")
    config.num_classes = dataset.num_classes
    config.batch_size = args.batch_size
    config.total_epoch = args.total_epochs
    config.hid_feats = args.hid_feats
    config.save_every = args.save_every
    config.graph_name = "mag240"
    config.feat = args.feat
    config.topo = args.topo
    config.fanouts = [20, 20, 20]
    config.global_in_feats = dataset.num_paper_features
    idx_split = dataset.get_idx_split()
    
    node_labels = torch.from_numpy(np.array(dataset.paper_label))
    node_labels = node_labels.flatten()
    torch.nan_to_num_(node_labels, nan=-1)
    node_labels = node_labels.type(torch.int64)
    print("start loading row and col")   
    row, col = dataset.edge_index("paper", "paper")
    print("start creating graph")  
    csr_topo = quiver.CSRTopo((torch.from_numpy(row), torch.from_numpy(col)))
    sampler = quiver.pyg.GraphSageSampler(csr_topo=csr_topo, sizes=config.fanouts, mode=config.topo)
    del row, col
    gc.collect()
    load_end = time.time()
    print(f"finish loading graph in {load_end - load_start}s")
    
    # DGL + Quiver Feature
    qfeat = None
    if args.mode != 1:
        quiver.init_p2p(device_list=list(range(world_size)))
        qfeat = quiver.Feature(0, device_list=list(range(world_size)), cache_policy="p2p_clique_replicate", device_cache_size='0G')
        gpu_parts = []
        cpu_part = np.array(list(range(node_labels.shape[0])))
        for i in range(world_size):
            gpu_parts.append(np.empty(0))
        
        device_config = quiver.feature.DeviceConfig(gpu_parts, cpu_part)
        qfeat.from_mmap(dataset.paper_feat, device_config)
        gc.collect()
        
    if args.mode == 0:
        mp.spawn(main_v0, args=(world_size, config, qfeat, sampler, node_labels, idx_split), nprocs=world_size, daemon=True)
    elif args.mode == 1:
        # DGL Only
        # feat is kept in CPU memory
        config.feat = "CPU"
        feat = torch.from_numpy(dataset.all_paper_feat).type(torch.float32)
        feat.share_memory_()
        mp.spawn(main_v1, args=(world_size, config, feat, sampler, node_labels, idx_split), nprocs=world_size, daemon=True)
        # Feature data is horizontally partitiones
    elif args.mode == 2:
        # P2 Data Vertical Split
        mp.spawn(main_v2, args=(world_size, config, qfeat, sampler, node_labels, idx_split), nprocs=world_size, daemon=True)
    elif args.mode == 3:
        # P3 Data Vertical Split + Intra-Model Parallelism
        mp.spawn(main_v3, args=(world_size, config, qfeat, sampler, node_labels, idx_split), nprocs=world_size, daemon=True)            