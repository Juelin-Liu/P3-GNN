import torch, quiver, dgl
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import time
import csv
from dataclasses import dataclass
from dgl import create_block


def partition_ids(rank: int, world_size: int, nids: torch.Tensor) -> torch.Tensor:
    step = int(nids.shape[0] / world_size)
    start_idx = rank * step
    end_idx = start_idx + step
    loc_ids = nids[start_idx : end_idx]
    return loc_ids.to(rank)

def print_model_weights(model: torch.nn.Module):
    for name, weight in model.named_parameters():
        if weight.requires_grad:
            print(name, weight, weight.shape, "\ngrad:", weight.grad)
        else:
            print(name, weight, weight.shape)


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
        return f"{round(size / 1000.0)} KB"
    elif size < 1e6:
        return f"{round(size / 1000.0)} KB"
    elif size < 1e9:
        return f"{round(size / 1000000.0)} MB"
    else:
        return f"{round(size / 1000000000.0)} GB"
    


# This function split the feature data horizontally
# each node's data is partitioned into 'world_size' chunks
# return the partition corresponding to the 'rank'
# Input args:
# rank: [0, world_size - 1]
# Output: feat
def get_local_feat(rank: int, world_size:int, feat: torch.Tensor, padding=True) -> torch.Tensor:
    org_feat_width = feat.shape[1]
    if padding and org_feat_width % world_size != 0:
        step = int(org_feat_width / world_size)
        pad = world_size - org_feat_width + step * world_size
        padded_width = org_feat_width + pad
        assert(padded_width % world_size == 0)
        step = int(padded_width / world_size)
        start_idx = rank * step
        end_idx = start_idx + step
        local_feat = None
        if rank == world_size - 1:
            # padding is required for P3 to work correctly
            local_feat = feat[:, start_idx : org_feat_width]
            zeros = torch.zeros((local_feat.shape[0], pad), dtype=local_feat.dtype)
            local_feat = torch.concatenate([local_feat, zeros], dim=1)
        else:
            local_feat = feat[:, start_idx : end_idx]
        return local_feat
    else:
        step = int(feat.shape[1] / world_size)
        start_idx = rank * step
        end_idx = min(start_idx + step, feat.shape[1])
        if rank == world_size - 1:
            end_idx = feat.shape[1]
        local_feat = feat[:, start_idx : end_idx]
        return local_feat
    
class QuiverGraphSageSampler():
    def __init__(self, sampler: quiver.pyg.GraphSageSampler):
        self.sampler = sampler
    
    def sample_dgl(self, seeds):
        """Sample k-hop neighbors from input_nodes

        Args:
            input_nodes (torch.LongTensor): seed nodes ids to sample from
        Returns:
            Tuple: Return results are the same with Dgl's sampler
            1. input_ndoes # to extract features
            2. output_nodes # to prefict label
            3. blocks # dgl blocks
        """
        self.sampler.lazy_init_quiver()
        adjs = []
        nodes = seeds

        for size in self.sampler.sizes:
            out, cnt = self.sampler.sample_layer(nodes, size)
            frontier, row_idx, col_idx = self.sampler.reindex(nodes, out, cnt)
            block = create_block(('coo', (col_idx, row_idx)), num_dst_nodes=nodes.shape[0], num_src_nodes=frontier.shape[0], device=self.sampler.device)
            adjs.append(block)
            nodes = frontier
        return nodes, seeds, adjs[::-1]
    
class QuiverDglSageSample():
    def __init__(self, 
                 rank: int,
                 world_size: int,
                 batch_size: int, 
                 nids:torch.Tensor, 
                 sampler: quiver.pyg.GraphSageSampler,
                 shuffle=True,
                 partition=True):
        self.rank = rank
        if partition:
            self.nids = partition_ids(rank, world_size, nids)
        else:
            self.nids = nids.to(rank) # train_nids
        self.cur_idx = 0
        self.max_idx = nids.shape[0]
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sampler = QuiverGraphSageSampler(sampler)     
        # self.sampler = sampler    

    def __iter__(self):
        self.cur_idx = 0
        if self.shuffle:
            dim = 0
            idx = torch.randperm(self.nids.shape[dim]).to(self.rank)
            self.nids = self.nids[idx]
        return self

    def __next__(self):
        if self.cur_idx < self.max_idx:
            seeds = self.nids[self.cur_idx : self.cur_idx + self.batch_size]
            self.cur_idx += self.batch_size
            return self.sampler.sample_dgl(seeds)
        else:
            raise StopIteration
                        
# class TrainProfiler:
#     def __init__(self, filepath: str) -> None:
#         self.items = []
#         self.path = filepath
#         self.fields = ["epoch", "val_acc", "epoch_time", "forward", "backward", "feat", "sample", "other"]        
    
#     def log_step_dict(self, item: dict):
#         for k, v in item.items():
#             if (type(v) == float):
#                 item[k] = round(v, 5)
#         self.items.append(item)
#         self.fields = list(item.keys())
        
#     def log_step(self, 
#                 epoch: int, 
#                 val_acc: float,
#                 epoch_time: float,
#                 forward: float,
#                 backward: float,
#                 feat: float,
#                 sample: float) -> dict:
        
#         other = epoch_time - forward - backward - feat - sample
#         item = {
#             "epoch": epoch,
#             "val_acc": val_acc,
#             "epoch_time": epoch_time,
#             "forward": forward,
#             "backward": backward,
#             "feat": feat,
#             "sample": sample,
#             "other": other
#         }

#         for k, v in item.items():
#             if (type(v) == type(1.0)):
#                 item[k] = round(v, 5)
#         self.items.append(item)
#         return item
    
#     def avg_epoch(self) -> float:
#         if (len(self.items) <= 1):
#             return 0
#         avg_epoch_time = 0.0
#         epoch = 0
#         for idx, item in enumerate(self.items):
#             if idx != 0:
#                 avg_epoch_time += item["epoch_time"]
#                 epoch += 1
#         return avg_epoch_time / epoch
    
    
#     def saveToDisk(self):
#         print("AVERAGE EPOCH TIME: ", round(self.avg_epoch(), 4))
#         with open(self.path, "w+") as file:
#             writer = csv.DictWriter(file, self.fields)
#             writer.writeheader()
#             for idx, item in enumerate(self.items):
#                 if idx > 0:
#                     writer.writerow(item)

# @dataclass
# class RunConfig:
#     rank: int = 0
#     world_size: int = 1
#     topo: str = "uva"
#     feat: str = "uva"
#     global_in_feats: int = -1
#     local_in_feats: int = -1
#     hid_feats: int = 128
#     num_classes: int = -1 # output feature size
#     batch_size: int = 1024
#     total_epoch: int = 30
#     save_every: int = 30
#     fanouts: list[int] = None
#     graph_name: str = "ogbn-arxiv"
#     log_path: str = "log.csv" # logging output path
#     checkpt_path: str = "checkpt.pt" # checkpt path
#     model: str = "sage" # model (sage or gat)
#     num_heads: int = 3 # if use GAT, number of heads in the model
#     mode: int = 1 # runner version
    
#     def uva_sample(self) -> bool:
#         return self.topo == 'uva'
    
#     def uva_feat(self) -> bool:
#         return self.feat == 'uva'
    
#     def set_logpath(self):
#         dir1 = f"{self.feat.lower()}feat"
#         dir2 = f"{self.topo.lower()}topo"
#         self.log_path = f"./logs/{self.graph_name}_v{self.mode}_w{self.world_size}_{dir1}_{dir2}_h{self.hid_feats}_b{self.batch_size}.csv"

import os, dgl, torch, time, csv, gc
from ogb.nodeproppred import DglNodePropPredDataset
import pandas as pd
import numpy as np

def _build_dgl_graph(indptr, indices, edges) -> dgl.DGLGraph:
    graph = dgl.graph(("csc", (indptr, indices, edges)))
    return graph

def preprocess(graph_name, in_dir, out_dir) -> None:
    out_dir = os.path.join(out_dir, graph_name)
    try:
        os.mkdir(out_dir)
    except Exception as e:
        print(e)
    
    id_type = torch.int64
    idtype_str = "64"
    dataset = DglNodePropPredDataset(graph_name, in_dir)
    graph = dataset[0][0]
    if graph_name == "ogbn-proteins":
        feat = graph.edata.pop("feat")
        torch.save(feat, os.path.join(out_dir, "feat.pt"))
        species = graph.ndata["species"]
        torch.save(species, os.path.join(out_dir, "species.pt"))
    else:
        feat: torch.Tensor = graph.dstdata.pop("feat")
        torch.save(feat, os.path.join(out_dir, "feat.pt"))
        del feat

    node_labels: torch.Tensor = dataset[0][1]
    node_labels = node_labels.flatten().clone()
    torch.nan_to_num_(node_labels, nan=0.0)
    node_labels: torch.Tensor = node_labels.type(torch.int64)
    
    torch.save(node_labels, os.path.join(out_dir, "label.pt"))

    idx_split = dataset.get_idx_split()
    train_idx = idx_split["train"].type(id_type)
    valid_idx = idx_split["valid"].type(id_type)
    test_idx = idx_split["test"].type(id_type)

    ntype = torch.zeros(graph.num_nodes(), dtype = torch.int64)
    count = 0
    for k in ["train", "valid", "test"]:
        ids = idx_split[k].type(id_type)
        ntype[ids] = count
        count  = count + 1
    torch.save(ntype, os.path.join(out_dir, f'ntype.pt'))


    torch.save(train_idx, os.path.join(out_dir, f"train_idx_{idtype_str}.pt"))
    torch.save(valid_idx, os.path.join(out_dir, f"valid_idx_{idtype_str}.pt"))
    torch.save(test_idx, os.path.join(out_dir, f"test_idx_{idtype_str}.pt"))

    indptr, indices, edges = graph.adj_tensors("csc")
    indptr = indptr.type(id_type)
    indices = indices.type(id_type)
    edges = edges.type(id_type)
    
    torch.save(indptr, os.path.join(out_dir, f"indptr_{idtype_str}.pt"))
    torch.save(indices, os.path.join(out_dir, f"indices_{idtype_str}.pt"))
    torch.save(edges, os.path.join(out_dir, f"edges_{idtype_str}.pt"))
    add_self_loop(out_dir, out_dir)

def add_self_loop(in_dir, out_dir=None):
    id_type = torch.int64
    idtype_str = "64"
    graph = load_dgl_graph(in_dir)
    graph = graph.remove_self_loop()
    graph = graph.add_self_loop()
    indptr, indices, edges = graph.adj_tensors("csc")
    indptr = indptr.type(id_type)
    indices = indices.type(id_type)
    edges = edges.type(id_type)
    if out_dir == None:
        out_dir = in_dir
    torch.save(indptr, os.path.join(out_dir, f"indptr_{idtype_str}_wsloop.pt"))
    torch.save(indices, os.path.join(out_dir, f"indices_{idtype_str}_wsloop.pt"))
    torch.save(edges, os.path.join(out_dir, f"edges_{idtype_str}_wsloop.pt"))
    
def load_graph(in_dir, is32=False, wsloop=False) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    idtype_str = "64"
    indptr = None
    indices = None
    if not wsloop: 
        indptr = torch.load(os.path.join(in_dir, f"indptr_{idtype_str}.pt"))
        indices = torch.load(os.path.join(in_dir, f"indices_{idtype_str}.pt"))
        # edges = torch.load(os.path.join(in_dir, f"edges_{idtype_str}.pt"))
        edges = torch.empty(0, dtype=indices.dtype)
    else:
        # with self loop
        indptr = torch.load(os.path.join(in_dir, f"indptr_{idtype_str}_wsloop.pt"))
        indices = torch.load(os.path.join(in_dir, f"indices_{idtype_str}_wsloop.pt"))
        # edges = torch.load(os.path.join(in_dir, f"edges_{idtype_str}_wsloop.pt"))
        edges = torch.empty(0, dtype=indices.dtype)
    if is32:
        return indptr.type(torch.int32), indices.type(torch.int32), edges.type(torch.int32)
    else:
        return indptr, indices, edges

def load_idx_split(in_dir, is32=False) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    idtype_str = "64"
    train_idx = torch.load(os.path.join(in_dir, f"train_idx_{idtype_str}.pt"))
    valid_idx = torch.load(os.path.join(in_dir, f"valid_idx_{idtype_str}.pt"))
    test_idx = torch.load(os.path.join(in_dir, f"test_idx_{idtype_str}.pt"))
    if is32:
        return train_idx.type(torch.int32), valid_idx.type(torch.int32), test_idx.type(torch.int32)
    else:
        return train_idx, valid_idx, test_idx

def load_feat_label(in_dir) -> (torch.Tensor, torch.Tensor, int):
    feat = torch.load(os.path.join(in_dir, f"feat.pt"))
    label = torch.load(os.path.join(in_dir, f"label.pt"))
    num_labels = torch.unique(label).shape[0]
    return feat, label, num_labels

def load_dgl_graph(in_dir, is32=False, wsloop=False) -> dgl.DGLGraph:
    indptr, indices, edges = load_graph(in_dir, is32, wsloop)
    graph = _build_dgl_graph(indptr, indices, edges)
    if is32:
        return graph.int()
    else:
        return graph
    
def get_dataset(graph_name, in_dir):
    dataset = DglNodePropPredDataset(graph_name, in_dir)
    return dataset

def get_metis_partition(in_dir, config, graph):
    assert config.partition_type in ["edge_balanced", "node_balanced", "random"]
    if config.partition_type == "random":
        return torch.randint(0, 4, (graph.num_nodes(),), dtype = torch.int32)
    if config.partition_type == "edge_balanced":
        edge_balanced = True
        return torch.load(f'{in_dir}/partition_map_{edge_balanced}').to(torch.int32)
    if config.partition_type == "node_balanced":
        edge_balanced = False
        return torch.load(f'{in_dir}/partition_map_{edge_balanced}').to(torch.int32)

def get_dgl_sampler(graph: dgl.DGLGraph, train_idx: torch.Tensor, graph_samler: dgl.dataloading.Sampler, system:str = "cpu", batch_size:int=1024, use_dpp=False) -> dgl.dataloading.dataloader.DataLoader:
    device = torch.cuda.current_device()
    dataloader = None
    drop_last = True
    
    if device == torch.cuda.device(0):
        print(f"before dataloader init graph formats: {graph.formats()}")

    if system == "cpu":
        dataloader = dgl.dataloading.DataLoader(
            graph=graph,               # The graph
            indices=train_idx,         # The node IDs to iterate over in minibatches
            graph_sampler=graph_samler,     # The neighbor sampler
            device="cpu",      # Put the sampled MFGs on CPU or GPU
            use_ddp=use_dpp, # enable ddp if using mutiple gpus
            # The following arguments are inherited from PyTorch DataLoader.
            batch_size=batch_size,    # Batch size
            shuffle=True,       # Whether to shuffle the nodes for every epoch
            drop_last=drop_last,    # Whether to drop the last incomplete batch
            use_uva=False,
            num_workers=1,
        )
    elif "uva" in system:
        graph.pin_memory_()
        assert(graph.is_pinned())
        dataloader = dgl.dataloading.DataLoader(
            graph=graph,               # The graph
            indices=train_idx.to(device),         # The node IDs to iterate over in minibatches
            graph_sampler=graph_samler,     # The neighbor sampler
            device=device,      # Put the sampled MFGs on CPU or GPU
            use_ddp=use_dpp, # enable ddp if using mutiple gpus
            # The following arguments are inherited from PyTorch DataLoader.
            batch_size=batch_size,    # Batch size
            shuffle=True,       # Whether to shuffle the nodes for every epoch
            drop_last=drop_last,    # Whether to drop the last incomplete batch
            use_uva=True,
            num_workers=0,
        )
    elif "gpu" in system:
        graph = graph.to(device)
        dataloader = dgl.dataloading.DataLoader(
            graph=graph,               # The graph
            indices=train_idx.to(device),         # The node IDs to iterate over in minibatches
            graph_sampler=graph_samler,     # The neighbor sampler
            device=device,      # Put the sampled MFGs on CPU or GPU
            use_ddp=use_dpp, # enable ddp if using mutiple gpus
            # The following arguments are inherited from PyTorch DataLoader.
            batch_size=batch_size,    # Batch size
            shuffle=True,       # Whether to shuffle the nodes for every epoch
            drop_last=drop_last,    # Whether to drop the last incomplete batch
            use_uva=False,
            num_workers=0,
        )
    if device == torch.cuda.device(0):
        print(f"after dataloader init graph formats: {graph.formats()}")
    return dataloader, graph

def get_memory_info(device=torch.cuda.current_device(), rd=0):
    allocated_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
    reserved_mb = torch.cuda.memory_reserved(device) / 1024 / 1024
    allocated_mb = round(allocated_mb, rd)
    reserved_mb = round(reserved_mb, rd)
    return allocated_mb, reserved_mb

class Timer:
    def __init__(self):
        self.start = time.time()
    def duration(self, rd=3):
        return round(time.time() - self.start, rd)
    def reset(self):
        self.start = time.time()

class CudaTimer:
    def __init__(self, stream=torch.cuda.current_stream()):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.stream = stream
        self.end_recorded = False
        self.start_event.record(stream=self.stream)

    def start(self):
        self.start_event.record(stream=self.stream)
        
    def end(self):
        self.end_event.record(stream=self.stream)
        self.end_recorded = True
        
    def reset(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.end_recorded = False
        
    def duration(self):
        assert(self.end_recorded)
        self.start_event.synchronize()
        self.end_event.synchronize()
        duration_ms = self.start_event.elapsed_time(self.end_event)
        duration_s = duration_ms / 1000
        return duration_s
    
class Config:
    def __init__(self, graph_name, world_size, num_epoch, fanouts,
                 batch_size, system, model, hid_size, cache_size, log_path, data_dir):
        try:
            self.machine_name = os.environ['MACHINE_NAME']
        except Exception as e:
            self.machine_name = "jupiter"
        self.graph_name = graph_name
        self.world_size = world_size
        self.num_epoch = num_epoch
        self.fanouts = fanouts
        self.batch_size = batch_size
        self.system = system
        self.model = model
        self.local_in_feats = -1
        self.global_in_feats = -1
        self.num_classes = -1
        self.cache_size = cache_size
        self.hid_size = hid_size
        self.log_path = log_path
        self.data_dir = data_dir
        self.num_redundant_layer = len(self.fanouts)
        self.partition_type = "random"

    def get_file_name(self):
        if "groot" not in self.system:
            return (f"{self.system}_{self.graph_name}_{self.model}_{self.batch_size}_{self.hid_size}_" + \
                     f"{len(self.fanouts)}x{self.fanouts[0]}_{self.cache_size}")
        else:
            return (f"{self.system}_{self.graph_name}_{self.model}_{self.batch_size}_{self.hid_size}_" + \
                    f"{len(self.fanouts)}x{self.fanouts[0]}_{self.num_redundant_layer}_{self.cache_size}")

    def header(self):
        return ["timestamp","machine_name", "graph_name", "world_size", "num_epoch", "fanouts", "num_redundant_layers", \
                "batch_size", "system", \
                    "model", "hid_size", "cache_size", "partition_type"]
    
    def content(self):
        return [  pd.Timestamp('now'), self.machine_name, self.graph_name, self.world_size, self.num_epoch, self.fanouts, self.num_redundant_layer, \
                    self.batch_size, self.system, self.model, self.hid_size, self.cache_size, self.partition_type]

    def __repr__(self):
        res = ""
        header = self.header()
        content = self.content()
        
        for header, ctn in zip(header, content):
            res += f"{header}={ctn} | "
        res += f"num_classes={self.num_classes}"
        res += "\n"
        return res    

class Profiler:
    def __init__(self, duration: float, sampling_time : float, feature_time: float,\
                 forward_time: float, backward_time: float, test_acc):
        self.duration = duration
        self.sampling_time = sampling_time
        self.feature_time = feature_time
        self.forward_time = forward_time
        self.backward_time = backward_time
        self.test_acc = test_acc
        self.allocated_mb, self.reserved_mb = get_memory_info()
        self.edges_computed = 0
        self.edge_skew = 0
        self.run_time = 0
    def header(self):
        header = ["duration (s)", "sampling (s)", "feature (s)", "forward (s)", "backward (s)",\
                    "allocated (MB)", "reserved (MB)", "test accuracy %", "edges_computed", "edge_skew", "run_time"]
        return header
    
    def content(self):
        content = [self.duration, self.sampling_time, self.feature_time, self.forward_time,\
                   self.backward_time, self.allocated_mb, self.reserved_mb, self.test_acc, \
                   self.edges_computed, self.edge_skew, self.run_time]
        return content
    
    def __repr__(self):
        res = ""
        header = self.header()
        content = self.content()
        for header, ctn in zip(header, content):
            res += f"{header}={ctn} | "
        res += "\n"
        return res

def empty_profiler():
    empty = -1
    profiler = Profiler(duration=empty, sampling_time=empty, feature_time=empty, forward_time=empty, backward_time=empty, test_acc=empty)
    return profiler

def oom_profiler():
    oom = "oom"
    profiler = Profiler(duration=oom, sampling_time=oom, feature_time=oom, forward_time=oom, backward_time=oom, test_acc=oom)
    return profiler


def get_duration(timers: list[CudaTimer], rb=3)->float:
    res = 0.0
    for timer in timers:
        res += timer.duration()
    return round(res, rb)

def write_to_csv(out_path, configs: list[Config], profilers: list[Profiler]):
    assert(len(configs) == len(profilers))
    def get_row(header, content):
        res = {}
        for k, v in zip(header, content):
            res[k] = v
        return res
    
    has_header = os.path.isfile(out_path)
    with open(out_path, 'a') as f:
        header = configs[0].header() + profilers[0].header()
        writer = csv.DictWriter(f, fieldnames=header)        
        if not has_header:
            writer.writeheader()
        for config, profiler in zip(configs, profilers):
            row = get_row(config.header() + profiler.header(), config.content() + profiler.content())
            writer.writerow(row)
    print("Experiment result has been written to: ", out_path)


def profile_edge_skew(edges_computed, profiler, rank, dist):
    profiler.edges_computed = sum(edges_computed)/len(edges_computed)
    edges_computed = sum(edges_computed)/len(edges_computed)
    edges_computed_max  = torch.tensor(edges_computed).to(rank)
    edges_computed_min  = torch.tensor(edges_computed).to(rank)
    edges_computed_avg  = torch.tensor(edges_computed).to(rank)
    dist.all_reduce(edges_computed_max, op = dist.ReduceOp.MAX)
    dist.all_reduce(edges_computed_min, op = dist.ReduceOp.MIN)
    dist.all_reduce(edges_computed_avg, op = dist.ReduceOp.SUM)
    profiler.edges_computed = edges_computed_avg.item()/4
    profiler.edge_skew = (edges_computed_max.item() - edges_computed_min.item()) / profiler.edges_computed

def metis(in_dir, graph_name):
    idtype_str = "64"
    SAVE_PATH = os.path.join(in_dir, graph_name)
    ws_self_loop = False
    indptr = torch.load( f'{SAVE_PATH}/indptr_{idtype_str}.pt')
    indices = torch.load(f'{SAVE_PATH}/indices_{idtype_str}.pt')
    edges = torch.empty(0, dtype=indices.dtype)
    graph = dgl.DGLGraph(('csc', (indptr,indices, edges)))
    ntype = torch.load(f'{SAVE_PATH}/ntype.pt')

    num_partitions = 4
    torch_type = torch.int32
    for edge_balanced  in [True, False]:
        partitions = dgl.metis_partition(graph, num_partitions, balance_ntypes=ntype, balance_edges = False)
        p_map = torch.zeros(graph.num_nodes(), dtype=torch_type)
        print(partitions)
        for p_id in partitions.keys():
            nodes = partitions[p_id].ndata['_ID']
            p_map[nodes] = p_id
            print(f"In partiiton {p_id}: nodes{nodes.shape}")
        p_map = p_map.to(torch_type)
        torch.save(p_map, f"{SAVE_PATH}/pmap_{edge_balanced}.pt")

def active_python():
    import psutil
    count = 0
    for i in psutil.process_iter():
        if "python" in i.name():
            count = count + 1
    return count == 1
