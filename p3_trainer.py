import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.cuda import Event as event
import time
from dgl.dataloading import DataLoader as DglDataLoader
from dgl import create_block
import csv
import torchmetrics.functional as MF
from utils import RunConfig
from models.p3_sage import ShuffleLayer

def print_model_weights(model: torch.nn.Module):
    for name, weight in model.named_parameters():
        if weight.requires_grad:
            print(name, weight, weight.shape, "\ngrad:", weight.grad)
        else:
            print(name, weight, weight.shape)
            
class TrainProfiler:
    def __init__(self, filepath: str) -> None:
        self.items = []
        self.path = filepath
        self.fields = ["epoch", "val_acc", "epoch_time", "forward", "backward", "feat", "sample", "other"]        
    
    def log_step(self, 
                epoch: int, 
                val_acc: float,
                epoch_time: float,
                forward: float,
                backward: float,
                feat: float,
                sample: float) -> dict:
        
        other = epoch_time - forward - backward - feat - sample
        item = {
            "epoch": epoch,
            "val_acc": val_acc,
            "epoch_time": epoch_time,
            "forward": forward,
            "backward": backward,
            "feat": feat,
            "sample": sample,
            "other": other
        }

        for k, v in item.items():
            if (type(v) == type(1.0)):
                item[k] = round(v, 5)
        self.items.append(item)
        return item
    
    def avg_epoch(self) -> float:
        if (len(self.items) <= 1):
            return 0
        avg_epoch_time = 0.0
        epoch = 0
        for idx, item in enumerate(self.items):
            if idx != 0:
                avg_epoch_time += item["epoch_time"]
                epoch += 1
        return avg_epoch_time / epoch
    
    
    def saveToDisk(self):
        print("AVERAGE EPOCH TIME: ", round(self.avg_epoch(), 4))
        with open(self.path, "w+") as file:
            writer = csv.DictWriter(file, self.fields)
            writer.writeheader()
            for idx, item in enumerate(self.items):
                if idx > 0:
                    writer.writerow(item)

class P3Trainer:
    def __init__(
        self,
        config: RunConfig,
        global_model: torch.nn.Module, # All Layers execpt for the first layer
        local_model: torch.nn.Module,
        train_data: DglDataLoader,
        val_data: DglDataLoader,
        local_feat: torch.Tensor,
        gloabl_feat: torch.Tensor, # assume stored in CPU memory
        label: torch.Tensor,
        global_optimizer: torch.optim.Optimizer,
        local_optimizer: torch.optim.Optimizer,
        nid_dtype: torch.dtype = torch.int64
    ) -> None:
        self.config = config
        self.rank = config.rank
        self.world_size = config.world_size
        self.device = torch.device(f"cuda:{self.rank}")
        self.local_feat = local_feat.to(self.device)
        self.global_feat = gloabl_feat
        self.node_labels = torch.flatten(label).to(self.device)
        self.train_data = train_data
        self.val_data = val_data
        self.gloabl_optimizer = global_optimizer
        self.local_optimizer = local_optimizer
        self.local_model = local_model.to(device=self.device)
        if config.world_size == 1:
            self.model = global_model.to(device=self.device)
        elif config.world_size > 1:
            self.model = DDP(global_model.to(device=self.device), device_ids=[self.rank], output_device=self.rank)
        self.num_classes = config.num_classes
        self.save_every = config.save_every        
        self.log = TrainProfiler(config.log_path)
        self.checkpt_path = config.checkpt_path
        # Initialize buffers for storing feature data fetched from other GPUs
        self.edge_size_lst: list = [(0, 0, 0)] * self.world_size
        self.est_node_size = self.config.batch_size * 20
        self.local_feat_width = self.local_feat.shape[1]
        self.input_node_buffer_lst: list[torch.Tensor] = [] # storing input nodes 
        self.input_feat_buffer_lst: list[torch.Tensor] = [] # storing input nodes 
        self.src_edge_buffer_lst: list[torch.Tensor] = [] # storing src nodes
        self.dst_edge_buffer_lst: list[torch.Tensor] = [] # storing dst nodes 
        self.global_grad_lst: list[torch.Tensor] = [] # storing feature data gathered for other gpus
        self.local_hid_buffer_lst: list[torch.Tensor] = [None] * self.world_size # storing feature data gathered from other gpus
        self.hid_feats = self.config.hid_feats
        for idx in range(self.world_size):
            self.input_node_buffer_lst.append(torch.zeros(self.est_node_size, dtype=nid_dtype, device=self.device))
            self.src_edge_buffer_lst.append(torch.zeros(self.est_node_size, dtype=nid_dtype, device=self.device))
            self.dst_edge_buffer_lst.append(torch.zeros(self.est_node_size, dtype=nid_dtype, device=self.device))
            self.global_grad_lst.append(torch.zeros([self.est_node_size, self.hid_feats], dtype=self.local_feat.dtype, device=self.device))
            self.input_feat_buffer_lst.append(torch.zeros([self.est_node_size, self.hid_feats], dtype=self.local_feat.dtype, device=self.device))

        self.stream = torch.cuda.current_stream(self.device)
        self.shuffle = ShuffleLayer.apply
    # fetch partial hid_feat from remote GPUs before forward pass
    # fetch partial gradient from remote GPUs during backward pass
    def _run_epoch_v3(self, epoch):
        forward = 0.0
        backward = 0.0
        sample_time = 0.0
        feat_time = 0.0
        concat_time = 0.0
        start = sample_start = time.time()
        iter_idx = 0
        for input_nodes, output_nodes, blocks in self.train_data:
            iter_idx += 1
            torch.cuda.synchronize(self.device)
            feat_start = sample_end = time.time()
            # 1. Send and Receive edges for all the other gpus
            src, dst = blocks[0].adj_sparse('coo')
            self.edge_size_lst[self.rank] = (self.rank, src.shape[0], input_nodes.shape[0]) # rank, edge_size, input_node_size
            dist.all_gather_object(object_list=self.edge_size_lst, obj=self.edge_size_lst[self.rank])
            
            for rank, edge_size, input_node_size in self.edge_size_lst:
                self.src_edge_buffer_lst[rank].resize_(edge_size)
                self.dst_edge_buffer_lst[rank].resize_(edge_size)
                self.input_node_buffer_lst[rank].resize_(input_node_size)
                # self.input_feat_buffer_lst[rank].resize_([input_node_size, self.local_feat_width])
                
            dist.all_gather(tensor_list=self.input_node_buffer_lst, tensor=input_nodes, async_op=False)
            dist.all_gather(tensor_list=self.src_edge_buffer_lst, tensor=src, async_op=False)
            dist.all_gather(tensor_list=self.dst_edge_buffer_lst, tensor=dst, async_op=False)
            for rank, _input_nodes in enumerate(self.input_node_buffer_lst):
                self.input_feat_buffer_lst[rank] = self.local_feat[_input_nodes]
            
            torch.cuda.synchronize(self.device)
            feat_end = forward_start = time.time()
            # 3. Fetch feature data and compute hid feature for other GPUs
            for r in range(self.world_size):
                input_nodes = self.input_node_buffer_lst[r]
                input_feats = self.input_feat_buffer_lst[r]
                block = None
                if r == self.rank:
                    block = blocks[0]
                else:
                    src = self.src_edge_buffer_lst[r]
                    dst = self.dst_edge_buffer_lst[r]
                    block = create_block(('coo', (src, dst)), device=self.device)
                                        
                self.local_hid_buffer_lst[r] = self.local_model(block, input_feats)
                self.global_grad_lst[r].resize_([block.num_dst_nodes(), self.hid_feats])
            
            local_hid: torch.Tensor = self.shuffle(self.rank, self.world_size, self.local_hid_buffer_lst[self.rank], self.local_hid_buffer_lst, self.global_grad_lst)
            output_labels = self.node_labels[output_nodes]
            # 6. Compute forward pass locally
            output_pred = self.model(blocks[1:], local_hid)            
            loss = F.cross_entropy(output_pred, output_labels) 
            # print(f"{self.rank=} {local_hid.grad=}")
            torch.cuda.synchronize(self.device)
            forward_end = backward_start = time.time()                
            # Backward Pass
            # TODO gather error gradients from other GPUs
            self.gloabl_optimizer.zero_grad()
            self.local_optimizer.zero_grad()

            loss.backward()
            self.gloabl_optimizer.step()
            self.local_optimizer.step()
            for r, global_grad in enumerate(self.global_grad_lst):
                if r != self.rank:
                    self.local_optimizer.zero_grad()
                    self.local_hid_buffer_lst[r].backward(global_grad)
                    self.local_optimizer.step()

            torch.cuda.synchronize(self.device)
            backward_end = time.time()

            forward += forward_end - forward_start
            backward += backward_end - backward_start
            feat_time += feat_end - feat_start
            sample_time += sample_end - sample_start
            # concat_time += concat_end - concat_start
            sample_start = time.time()

        
        torch.cuda.synchronize(self.device)
        end = time.time()
        epoch_time = end - start
        acc = self.evaluate_v3()
        if self.rank == 0 or self.world_size == 1:
            info = self.log.log_step(epoch, acc, epoch_time, forward, backward, feat_time, sample_time)
            print(info)
            
    def _save_checkpoint(self, epoch):
        if self.rank == 0 or self.world_size == 1:
            ckp = None
            if self.world_size == 1:
                ckp = self.model.state_dict()
            elif self.world_size > 1: 
                # using ddp
                ckp = self.model.module.state_dict()
            torch.save(ckp, self.checkpt_path)
            print(f"Epoch {epoch} | Training checkpoint saved at {self.checkpt_path}")

    def train_v3(self):
        self.model.train()
        for epoch in range(self.config.total_epoch):
            self._run_epoch_v3(epoch)
            if self.rank == 0 or self.world_size == 1:
                self.log.saveToDisk()
                if epoch % self.save_every == 0 and epoch > 0:
                    self._save_checkpoint(epoch)
                        
    def evaluate_v3(self):
        self.model.eval()
        ys = []
        y_hats = []
        for it, (input_nodes, output_nodes, blocks) in enumerate(self.val_data):
            with torch.no_grad():
                # 1. Send and Receive edges for all the other gpus
                src, dst = blocks[0].adj_sparse('coo')
                self.edge_size_lst[self.rank] = (self.rank, src.shape[0], input_nodes.shape[0]) # rank, edge_size, input_node_size
                dist.all_gather_object(object_list=self.edge_size_lst, obj=self.edge_size_lst[self.rank])
                for rank, edge_size, input_node_size in self.edge_size_lst:
                    self.src_edge_buffer_lst[rank].resize_(edge_size)
                    self.dst_edge_buffer_lst[rank].resize_(edge_size)
                    self.input_node_buffer_lst[rank].resize_(input_node_size)

                dist.all_gather(tensor_list=self.input_node_buffer_lst, tensor=input_nodes)
                dist.all_gather(tensor_list=self.src_edge_buffer_lst, tensor=src)
                dist.all_gather(tensor_list=self.dst_edge_buffer_lst, tensor=dst)
                
                local_hid = None
                # 3. Fetch feature data and compute hid feature for other GPUs
                for r in range(self.world_size):
                    input_nodes = self.input_node_buffer_lst[r]
                    input_feats = self.local_feat[input_nodes]
                    block = None
                    if r == self.rank:
                        block = blocks[0]
                    else:
                        src = self.src_edge_buffer_lst[r]
                        dst = self.dst_edge_buffer_lst[r]
                        block = create_block(('coo', (src, dst)), device=self.device)
                                            
                    self.local_hid_buffer_lst[r] = self.local_model(block, input_feats)
                    self.global_grad_lst[r].resize_([block.num_dst_nodes(), self.hid_feats])
                local_hid = self.shuffle(self.rank, self.world_size, self.local_hid_buffer_lst[self.rank], self.local_hid_buffer_lst, self.global_grad_lst)     
                ys.append(self.node_labels[output_nodes])
                y_hats.append(self.model(blocks[1:], local_hid))
                
        acc = MF.accuracy(
            torch.cat(y_hats),
            torch.cat(ys),
            task="multiclass",
            num_classes=self.num_classes)
    
        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        return (acc / self.world_size).item()