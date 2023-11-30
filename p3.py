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
from ogb.nodeproppred import DglNodePropPredDataset
from models.sage import Sage, create_sage_p3
from models.gat import Gat, create_gat_p3
from p3_trainer_v2 import P3Trainer
import gc
from torch.distributed import init_process_group, destroy_process_group, barrier
import os
import torch.multiprocessing as mp
from dgl.utils import pin_memory_inplace
from p3lib.utils import *

def get_dgl_dataloader(rank: int, config: Config,
                         graph: dgl.DGLGraph, 
                         train_nids: torch.Tensor,
                         use_dpp=True,
                         use_uva=False) -> dgl.dataloading.dataloader.DataLoader:
    device = torch.device(f"cuda:{rank}")
    if "gpu" in config.system:
        graph = graph.to(device)
    dataloader = dgl.dataloading.DataLoader(
        # The following arguments are specific to DGL's DataLoader.
        graph=graph,              # The graph
        indices=train_nids.to(device),         # The node IDs to iterate over in minibatches
        graph_sampler=dgl.dataloading.NeighborSampler(config.fanouts)
,            # The neighbor sampler
        device=device,      # Put the sampled MFGs on CPU or GPU
        use_ddp=use_dpp, # enable ddp if using mutiple gpus
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=config.batch_size,    # Batch size
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=True,    # Whether to drop the last incomplete batch
        num_workers=0,       # Number of sampler processes
        use_uva=use_uva
    )
    return dataloader, graph


def create_p3_model(rank: int, config: Config, num_heads=4):
    if config.model == 'sage':
        return create_sage_p3(rank, config.local_in_feats, hid_feats=config.hid_size, num_layers=len(config.fanouts), num_classes=config.num_classes)
    elif config.model == 'gat':
        return create_gat_p3(rank, config.local_in_feats, hid_feats=config.hid_size, num_layers=len(config.fanouts), num_classes=config.num_classes, num_heads=num_heads)
    
    
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

def train_p3(rank:int, 
         in_dir: str,
         config: Config,
         graph: dgl.DGLGraph,
         node_labels: torch.Tensor, 
         train_idx: torch.Tensor,
         valid_idx: torch.Tensor):
    ddp_setup(rank, config.world_size)
    node_labels = node_labels.to(rank)
    feat = load_p3_feat(in_dir, rank, config.world_size)
    config.global_in_feats = feat.shape[1] * config.world_size
    config.local_in_feats = feat.shape[1]
    pinned_handle = None
    if "uva" in config.system:
        pinned_handle = pin_memory_inplace(feat)
    else:
        feat = feat.to(rank)
        graph = graph.to(rank)

    print(f"{rank=} create models")
    local_model, global_model = create_p3_model(rank, config)                                           
    train_dataloader, graph = get_dgl_dataloader(rank, config, graph, train_idx, use_dpp=True, use_uva="uva" in config.system)
    val_dataloader, graph = get_dgl_dataloader(rank, config, graph, valid_idx, use_dpp=True, use_uva="uva" in config.system)
    global_optimizer = torch.optim.Adam(global_model.parameters(), lr=1e-3)
    local_optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-3)
    print(f"{rank=} init trainer")
    trainer = P3Trainer(rank, config, global_model, local_model, train_dataloader, val_dataloader, feat, node_labels, global_optimizer, local_optimizer, nid_dtype=torch.int32)
    print(f"{rank=} start training")
    trainer.train()
    trainer.evaluate()
    trainer.log()
    destroy_process_group()

def get_configs(graph_name, system, log_path, data_dir):
    fanouts = [[20, 20, 20]]
    batch_sizes = [256]
    models = ["sage","gat"]
    hid_sizes = [256]
    cache_sizes = [0]
    # fanouts = [[20,20,20],[20,20,20,20], [30,30,30]]
    # batch_sizes = [1024, 4096]
    # models = ['gat', 'sage']
    # hid_sizes = [256, 512]
    configs: list[config] = []
    for fanout in fanouts:
        for batch_size in batch_sizes:
            for model in models:
                for hid_size in hid_sizes:
                    for cache_size in cache_sizes:
                        config = Config(graph_name=graph_name, 
                                        world_size=4, 
                                        num_epoch=5, 
                                        fanouts=fanout, 
                                        batch_size=batch_size, 
                                        system=system, 
                                        model=model,
                                        hid_size=hid_size, 
                                        cache_size=cache_size, 
                                        log_path=log_path,
                                        data_dir=data_dir)
                        configs.append(config)
    return configs

def bench_p3_batch(configs: list[Config]):
    for config in configs:
        if config.graph_name == "ogbn-products" or config.graph_name == "com-orkut":
            config.system = "p3-gpu"
        if config.graph_name == "ogbn-papers100M" or config.graph_name== "com-friendster":
            config.system = "p3-uva"
            assert(config.graph_name == configs[0].graph_name)
    
    in_dir = os.path.join(config.data_dir, config.graph_name)
    prep_p3_feat(in_dir, config.world_size)
    label, num_label = load_label(in_dir)
    graph = load_dgl_graph(in_dir, is32=True, wsloop=True)
    graph.create_formats_()
    graph.pin_memory_()
    train_idx, valid_idx, test_idx = load_idx_split(in_dir, is32=True)
    # feats = [None] * config.world_size
    # for i in range(config.world_size):
    #     feats[i] = get_local_feat(i, config.world_size, feat, padding=True).clone()
    #     if i == 0:
    #         global_in_feats = feats[i].shape[1] * config.world_size
    #         local_in_feats = feats[i].shape[1]
    #     assert(global_in_feats == feats[i].shape[1] * config.world_size)
    #     assert(local_in_feats == feats[i].shape[1])
    # del feat
        
    for config in configs:
        config.num_classes = num_label
        # config.global_in_feats = global_in_feats
        # config.local_in_feats = local_in_feats
        try:       
            mp.spawn(train_p3, args=(in_dir, config, graph, label, train_idx, valid_idx), nprocs=config.world_size, daemon=True)            
        except Exception as e:
            print(e, "exception")
            # if "out of memory"in str(e):
            #     print("oom config", config)
            # else:
            #     write_to_csv(config.log_path, [config], [empty_profiler()])
            #     with open(f"exceptions/{config.get_file_name()}" , 'w') as fp:
            #         fp.write(str(e))
        
    
if __name__ == "__main__":
    root_dir = in_dir = "/data/snap/"
    configs = get_configs("com-friendster", "p3-uva", "/home/juelin/P3-GNN/p3.csv", in_dir)
    bench_p3_batch(configs=configs)