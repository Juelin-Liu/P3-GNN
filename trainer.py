import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.cuda import Event as event
import time
from dgl.dataloading import DataLoader as DglDataLoader
import csv
import torchmetrics.functional as MF
from utils import RunConfig

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

class Trainer:
    def __init__(
        self,
        config: RunConfig,
        model: torch.nn.Module,
        train_data: DglDataLoader,
        val_data: DglDataLoader,
        local_feat: torch.Tensor,
        gloabl_feat: torch.Tensor, # assume stored in CPU memory
        label: torch.Tensor,
        optimizer: torch.optim.Optimizer,
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
        self.optimizer = optimizer
        if config.world_size == 1:
            self.model = model.to(device=self.device)
        elif config.world_size > 1:
            self.model = DDP(model.to(device=self.device), device_ids=[self.rank], output_device=self.rank)
        self.num_classes = config.num_classes
        self.save_every = config.save_every        
        self.log = TrainProfiler(config.log_path)
        self.checkpt_path = config.checkpt_path
        # Initialize buffers for storing feature data fetched from other GPUs
        self.input_node_size_lst: list= [(0, 0)] * self.world_size
        self.est_node_size = self.config.batch_size * 20
        self.local_feat_width = self.local_feat.shape[1]
        self.input_node_buffer_lst: list[torch.Tensor] = [] # storing input node for gathering feature data
        self.global_feat_buffer_lst: list[torch.Tensor] = [] # storing feature data gathered for other gpus
        self.local_feat_buffer_lst: list[torch.Tensor] = [] # storing feature data gathered from other gpus
        for idx in range(self.world_size):
            self.input_node_buffer_lst.append(torch.zeros(self.est_node_size, dtype=nid_dtype, device=self.device))
            self.global_feat_buffer_lst.append(torch.zeros([self.est_node_size, self.local_feat_width], dtype=self.local_feat.dtype, device=self.device))
            self.local_feat_buffer_lst.append(torch.zeros([self.est_node_size, self.local_feat_width], dtype=self.local_feat.dtype, device=self.device))
        # torch.cuda.set_device(self.device)
        self.stream = torch.cuda.current_stream(self.device)
    
    def _run_epoch_v1(self, epoch): 
        forward = 0.0
        backward = 0.0
        sample_time = 0.0
        feat_time = 0.0
        
        start = time.time()
        sample_start = time.time()
        for input_nodes, output_nodes, blocks in self.train_data:
            torch.cuda.synchronize(self.device)
            feat_start = sample_end = time.time()
            input_feats = self.local_feat[input_nodes]
            output_labels = self.node_labels[output_nodes]

            torch.cuda.synchronize(self.device)
            feat_end = forward_start = time.time()
            output_pred = self.model(blocks, input_feats)            
            loss = F.cross_entropy(output_pred, output_labels)

            torch.cuda.synchronize(self.device)
            forward_end = backward_start = time.time()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            torch.cuda.synchronize(self.device)
            backward_end = time.time()
            forward += forward_end - forward_start
            backward += backward_end - backward_start
            feat_time += feat_end - feat_start
            sample_time += sample_end - sample_start
            
            sample_start = time.time()
        end = time.time()
        epoch_time = end - start
        if self.rank == 0 or self.world_size == 1:
            acc = self.evaluate()
            info = self.log.log_step(epoch, acc, epoch_time, forward, backward, feat_time, sample_time)
            print(info)

            
    # def _run_epoch_v1(self, epoch):
    #     forward = 0.0
    #     backward = 0.0
    #     sample_time = 0.0
    #     feat_time = 0.0
        
    #     start = event(enable_timing=True, interprocess=True)
    #     end = event(enable_timing=True, interprocess=True)
    #     sample_start = event(enable_timing=True, interprocess=True)
    #     sample_end = event(enable_timing=True, interprocess=True)
    #     feat_start = event(enable_timing=True, interprocess=True)
    #     feat_end = event(enable_timing=True, interprocess=True)        
    #     forward_start = event(enable_timing=True, interprocess=True)
    #     forward_end = event(enable_timing=True, interprocess=True)
    #     backward_start = event(enable_timing=True, interprocess=True)
    #     backward_end = event(enable_timing=True, interprocess=True)
        
    #     # stream = torch.cuda.current_stream(device=self.device)
    #     stream = self.stream
    #     start.record(stream)
    #     sample_start.record(stream)
    #     iter = 0
    #     for input_nodes, output_nodes, blocks in self.train_data:
    #         iter+=1
    #         print(f"rank={self.rank} {iter=} {forward=} {backward=} {sample_time=} {feat_time=}")
    #         sample_end.record(stream)
    #         feat_start.record(stream)
    #         input_feats = self.local_feat[input_nodes]
    #         output_labels = self.node_labels[output_nodes]
    #         feat_end.record(stream)
    #         forward_start.record(stream)
    #         output_pred = self.model(blocks, input_feats)            
    #         loss = F.cross_entropy(output_pred, output_labels)
    #         forward_end.record(stream)
    #         backward_start.record(stream)
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
    #         backward_end.record(stream)
    #         torch.cuda.synchronize()
    #         forward += forward_start.elapsed_time(forward_end)
    #         backward += backward_start.elapsed_time(backward_end)
    #         feat_time += feat_start.elapsed_time(feat_end)
    #         sample_time += sample_start.elapsed_time(sample_end)
    #         sample_start.record(stream)
    #     end.record(stream)
    #     epoch_time = start.elapsed_time(end)
        
    #     if self.rank == 0 or self.world_size == 1:
    #         acc = self.evaluate()
    #         info = self.log.log_step(epoch, acc, epoch_time, forward, backward, feat_time, sample_time)
    #         print(info)

    # fetch data from remote GPUs before forward pass
    def _run_epoch_v2(self, epoch):
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
            # 1. Send and Receive input_nodes for all the other gpus
            self.input_node_size_lst[self.rank] = (self.rank, input_nodes.shape[0])
            dist.all_gather_object(object_list=self.input_node_size_lst, obj=self.input_node_size_lst[self.rank])
            for rank, input_node_size in self.input_node_size_lst:
                self.input_node_buffer_lst[rank].resize_(input_node_size)
                self.global_feat_buffer_lst[rank].resize_([input_node_size, self.local_feat_width])
                self.local_feat_buffer_lst[rank].resize_([input_nodes.shape[0], self.local_feat_width]) # 
                
            dist.all_gather(tensor_list=self.input_node_buffer_lst, tensor=input_nodes)
            # 3. Fetch feature data for other GPUs
            for rank in range(self.world_size):
                self.global_feat_buffer_lst[rank][:] = self.local_feat[self.input_node_buffer_lst[rank]][:]
            
            # 4. Send & Receive feature data from other GPUs
            for rank in range(self.world_size):
                if rank == self.rank:
                    dist.gather(tensor=self.global_feat_buffer_lst[rank], gather_list=self.local_feat_buffer_lst, dst=rank, async_op=False) # gathering data from other GPUs
                else:
                    dist.gather(tensor=self.global_feat_buffer_lst[rank], gather_list=None, dst=rank, async_op=False) # gathering data from other GPUs
            
            torch.cuda.synchronize(self.device)
            concat_start = time.time()
            input_feats = torch.cat(self.local_feat_buffer_lst, dim=1)
            torch.cuda.synchronize(self.device)
            concat_end = time.time()
                        
            output_labels = self.node_labels[output_nodes]
            
            torch.cuda.synchronize(self.device)
            feat_end = forward_start = time.time()
            # 6. Compute forward pass locally
            output_pred = self.model(blocks, input_feats)            
            loss = F.cross_entropy(output_pred, output_labels) 
                
            torch.cuda.synchronize(self.device)
            forward_end = backward_start = time.time()                
            # Backward Pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            torch.cuda.synchronize(self.device)
            backward_end = time.time()
            
            forward += forward_end - forward_start
            backward += backward_end - backward_start
            feat_time += feat_end - feat_start
            sample_time += sample_end - sample_start
            # concat_time += concat_end - concat_start
            torch.cuda.synchronize(self.device)
            sample_start = time.time()
        
        torch.cuda.synchronize(self.device)
        end = time.time()
        epoch_time = end - start
        if self.rank == 0 or self.world_size == 1:
            acc = self.evaluate()
            info = self.log.log_step(epoch, acc, epoch_time, forward, backward, feat_time, sample_time)
            print(info, "concat:", round(concat_time, 4))
            
    # # fetch partial hid_feat from remote GPUs before forward pass
    # # fetch partial gradient from remote GPUs during backward pass
    # def _run_epoch_v3(self, epoch):
    #     forward = 0.0
    #     backward = 0.0
    #     sample_time = 0.0
    #     feat_time = 0.0
    #     concat_time = 0.0
    #     start = sample_start = time.time()
    #     iter_idx = 0
    #     for input_nodes, output_nodes, blocks in self.train_data:
    #         iter_idx += 1
    #         torch.cuda.synchronize(self.device)
    #         feat_start = sample_end = time.time()
    #         # 1. Send and Receive input_nodes for all the other gpus
    #         self.input_node_size_lst[self.rank] = (self.rank, input_nodes.shape[0])
    #         dist.all_gather_object(object_list=self.input_node_size_lst, obj=self.input_node_size_lst[self.rank])
    #         for rank, input_node_size in self.input_node_size_lst:
    #             self.input_node_buffer_lst[rank].resize_(input_node_size)
    #             self.global_feat_buffer_lst[rank].resize_([input_node_size, self.local_feat_width])
    #             self.local_feat_buffer_lst[rank].resize_([input_nodes.shape[0], self.local_feat_width]) # 
                
    #         dist.all_gather(tensor_list=self.input_node_buffer_lst, tensor=input_nodes)
    #         # 3. Fetch feature data for other GPUs
    #         for rank in range(self.world_size):
    #             self.global_feat_buffer_lst[rank][:] = self.local_feat[self.input_node_buffer_lst[rank]][:]
            
    #         # 4. Send & Receive feature data from other GPUs
    #         for rank in range(self.world_size):
    #             if rank == self.rank:
    #                 dist.gather(tensor=self.global_feat_buffer_lst[rank], gather_list=self.local_feat_buffer_lst, dst=rank, async_op=False) # gathering data from other GPUs
    #             else:
    #                 dist.gather(tensor=self.global_feat_buffer_lst[rank], gather_list=None, dst=rank, async_op=False) # gathering data from other GPUs
            
    #         torch.cuda.synchronize(self.device)
    #         concat_start = time.time()
    #         input_feats = torch.cat(self.local_feat_buffer_lst, dim=1)
    #         torch.cuda.synchronize(self.device)
    #         concat_end = time.time()
                        
    #         output_labels = self.node_labels[output_nodes]
            
    #         torch.cuda.synchronize(self.device)
    #         feat_end = forward_start = time.time()
    #         # 6. Compute forward pass locally
    #         output_pred = self.model(blocks, input_feats)            
    #         loss = F.cross_entropy(output_pred, output_labels) 
                
    #         torch.cuda.synchronize(self.device)
    #         forward_end = backward_start = time.time()                
    #         # Backward Pass
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
            
    #         torch.cuda.synchronize(self.device)
    #         backward_end = time.time()
            
    #         forward += forward_end - forward_start
    #         backward += backward_end - backward_start
    #         feat_time += feat_end - feat_start
    #         sample_time += sample_end - sample_start
    #         # concat_time += concat_end - concat_start
    #         torch.cuda.synchronize(self.device)
    #         sample_start = time.time()
        
    #     torch.cuda.synchronize(self.device)
    #     end = time.time()
    #     epoch_time = end - start
    #     if self.rank == 0 or self.world_size == 1:
    #         acc = self.evaluate()
    #         info = self.log.log_step(epoch, acc, epoch_time, forward, backward, feat_time, sample_time)
    #         print(info, "concat:", round(concat_time, 4))

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

    def train_v1(self):
        self.model.train()
        for epoch in range(self.config.total_epoch):
            self._run_epoch_v1(epoch)
            if self.rank == 0 or self.world_size == 1:
                self.log.saveToDisk()
                if epoch % self.save_every == 0 and epoch > 0:
                    self._save_checkpoint(epoch)
    
    def train_v2(self):
        self.model.train()
        for epoch in range(self.config.total_epoch):
            self._run_epoch_v2(epoch)
            if self.rank == 0 or self.world_size == 1:
                self.log.saveToDisk()
                if epoch % self.save_every == 0 and epoch > 0:
                    self._save_checkpoint(epoch)
                        
    def evaluate(self):
        self.model.eval()
        ys = []
        y_hats = []
        for it, (input_nodes, output_nodes, blocks) in enumerate(self.val_data):
            with torch.no_grad():
                if self.config.mode == 1:
                    x = self.local_feat[input_nodes]
                    ys.append(self.node_labels[output_nodes])
                    y_hats.append(self.model(blocks, x))
                else:
                    input_nodes=input_nodes.to("cpu")
                    x = self.global_feat[input_nodes].to(self.device)
                    ys.append(self.node_labels[output_nodes])
                    y_hats.append(self.model(blocks, x))
        return MF.accuracy(
            torch.cat(y_hats),
            torch.cat(ys),
            task="multiclass",
            num_classes=self.num_classes).item()
    