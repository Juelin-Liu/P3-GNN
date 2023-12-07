import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
from dgl.dataloading import DataLoader as DglDataLoader
import torchmetrics.functional as MF
from utils import RunConfig, TrainProfiler, QuiverDglSageSample
import quiver
from dgl.utils import gather_pinned_tensor_rows

class P2Trainer:
    def __init__(
        self,
        config: RunConfig,
        model: torch.nn.Module,
        train_data: DglDataLoader,
        val_data: DglDataLoader,
        local_feat: torch.Tensor, # local feature
        label: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        nid_dtype: torch.dtype = torch.int32
    ) -> None:
        self.config = config
        self.rank = config.rank
        self.world_size = config.world_size
        self.device = torch.device(f"cuda:{self.rank}")
        self.local_feat = local_feat # horizontally partitioned feature
        self.node_labels = label
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        if config.world_size == 1:
            self.model = model
        elif config.world_size > 1:
            self.model = DDP(model, device_ids=[self.rank], output_device=self.rank)
        self.num_classes = config.num_classes
        self.save_every = config.save_every
        self.feat_mode = config.feat    
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
            self.global_feat_buffer_lst.append(torch.zeros([self.est_node_size, self.local_feat_width], dtype=torch.float32, device=self.device))
            self.local_feat_buffer_lst.append(torch.zeros([self.est_node_size, self.local_feat_width], dtype=torch.float32, device=self.device))
        # torch.cuda.set_device(self.device)
        self.stream = torch.cuda.current_stream(self.device)

    # fetch data from remote GPUs before forward pass
    def _run_epoch(self, epoch):
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
                # self.global_feat_buffer_lst[rank].resize_([input_node_size, self.local_feat_width])
                self.local_feat_buffer_lst[rank].resize_([input_nodes.shape[0], self.local_feat_width]) # 
            
            dist.all_gather(tensor_list=self.input_node_buffer_lst, tensor=input_nodes)
            # 3. Fetch feature data for other GPUs
            # for rank in range(self.world_size):
            #     self.global_feat_buffer_lst[rank] = self.local_feat[self.input_node_buffer_lst[rank]]
            #     torch.cuda.synchronize(self.device)                
            #     print(f"feat {self.rank=} {rank=}")
                
            for rank, _input_nodes in enumerate(self.input_node_buffer_lst):
                if self.feat_mode == 'gpu':
                    self.global_feat_buffer_lst[rank] = self.local_feat[_input_nodes]
                elif self.feat_mode == 'uva':
                    self.global_feat_buffer_lst[rank] = gather_pinned_tensor_rows(self.local_feat, _input_nodes)
                else: # 'cpu'
                    self.global_feat_buffer_lst[rank] = self.local_feat[_input_nodes.to('cpu')].to(self.device)
                    
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
            concat_time += concat_end - concat_start
            torch.cuda.synchronize(self.device)
            sample_start = time.time()
        
        torch.cuda.synchronize(self.device)
        end = time.time()
        epoch_time = end - start
        acc = self.evaluate()
        if self.rank == 0 or self.world_size == 1:
            info = self.log.log_step(epoch, acc, epoch_time, forward, backward, feat_time, sample_time)
            print(info, "concat:", round(concat_time, 4))

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

    def train(self):
        self.model.train()
        for epoch in range(self.config.total_epoch):
            self._run_epoch(epoch)
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
                # 1. Send and Receive input_nodes for all the other gpus
                self.input_node_size_lst[self.rank] = (self.rank, input_nodes.shape[0])
                dist.all_gather_object(object_list=self.input_node_size_lst, obj=self.input_node_size_lst[self.rank])
                for rank, input_node_size in self.input_node_size_lst:
                    self.input_node_buffer_lst[rank].resize_(input_node_size)
                    # self.global_feat_buffer_lst[rank].resize_([input_node_size, self.local_feat_width])
                    self.local_feat_buffer_lst[rank].resize_([input_nodes.shape[0], self.local_feat_width]) # 
                    
                dist.all_gather(tensor_list=self.input_node_buffer_lst, tensor=input_nodes)
                # 3. Fetch feature data for other GPUs
                for rank, _input_nodes in enumerate(self.input_node_buffer_lst):
                    if self.feat_mode == 'gpu':
                        self.global_feat_buffer_lst[rank] = self.local_feat[_input_nodes]
                    elif self.feat_mode == 'uva':
                        self.global_feat_buffer_lst[rank] = gather_pinned_tensor_rows(self.local_feat, _input_nodes)
                    else: # 'cpu'
                        self.global_feat_buffer_lst[rank] = self.local_feat[_input_nodes.to('cpu')].to(self.device)
                        
                # 4. Send & Receive feature data from other GPUs
                for rank in range(self.world_size):
                    if rank == self.rank:
                        dist.gather(tensor=self.global_feat_buffer_lst[rank], gather_list=self.local_feat_buffer_lst, dst=rank, async_op=False) # gathering data from other GPUs
                    else:
                        dist.gather(tensor=self.global_feat_buffer_lst[rank], gather_list=None, dst=rank, async_op=False) # gathering data from other GPUs
                x = torch.cat(self.local_feat_buffer_lst, dim=1)
                ys.append(self.node_labels[output_nodes])
                y_hats.append(self.model(blocks, x))
                
        acc = MF.accuracy(
            torch.cat(y_hats),
            torch.cat(ys),
            task="multiclass",
            num_classes=self.num_classes)
        
        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        return (acc / self.world_size).item()
    