import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
from dgl.dataloading import DataLoader as DglDataLoader
from dgl import create_block
from dgl.utils import gather_pinned_tensor_rows
import torchmetrics.functional as MF
from p3lib.utils import *
from models.sage import *

class P3Trainer:
    def __init__(
        self,
        rank: int,
        config: Config,
        global_model: torch.nn.Module, # All Layers execpt for the first layer
        local_model: torch.nn.Module,
        train_data: DglDataLoader,
        val_data: DglDataLoader,
        local_feat: torch.Tensor, # Support UVA / GPU Feature Extraction
        node_labels: torch.Tensor,
        global_optimizer: torch.optim.Optimizer,
        local_optimizer: torch.optim.Optimizer,
        nid_dtype: torch.dtype = torch.int32
    ) -> None:
        self.rank = rank
        self.config = config
        self.world_size = config.world_size
        self.device = torch.device(f"cuda:{self.rank}")
        self.local_feat = local_feat
        self.node_labels = node_labels
        self.train_data = train_data
        self.val_data = val_data
        self.gloabl_optimizer = global_optimizer
        self.local_optimizer = local_optimizer
        self.local_model = local_model
        self.test_acc = 0
        if config.world_size == 1:
            self.model = global_model
        elif config.world_size > 1:
            self.model = DDP(global_model, device_ids=[self.rank], output_device=self.rank)
        self.num_classes = config.num_classes
        # Initialize buffers for storing feature data fetched from other GPUs
        self.edge_size_lst: list = [(0, 0, 0, 0)] * self.world_size #(rank, num_edges, num_dst_nodes, num_src_nodes)
        self.est_node_size = self.config.batch_size * 20
        self.local_feat_width = self.local_feat.shape[1]
        self.input_node_buffer_lst: list[torch.Tensor] = [] # storing input nodes 
        self.input_feat_buffer_lst: list[torch.Tensor] = [] # storing input nodes 
        self.src_edge_buffer_lst: list[torch.Tensor] = [] # storing src nodes
        self.dst_edge_buffer_lst: list[torch.Tensor] = [] # storing dst nodes 
        self.global_grad_lst: list[torch.Tensor] = [] # storing feature data gathered for other gpus
        self.local_hid_buffer_lst: list[torch.Tensor] = [None] * self.world_size # storing feature data gathered from other gpus
        self.hid_feats = self.config.hid_size
        for i in range(self.world_size):
            self.input_node_buffer_lst.append(torch.zeros(self.est_node_size, dtype=nid_dtype, device=self.device))
            self.src_edge_buffer_lst.append(torch.zeros(self.est_node_size, dtype=nid_dtype, device=self.device))
            self.dst_edge_buffer_lst.append(torch.zeros(self.est_node_size, dtype=nid_dtype, device=self.device))
            self.global_grad_lst.append(torch.zeros([self.est_node_size, self.hid_feats], dtype=torch.float32, device=self.device))
            self.input_feat_buffer_lst.append(torch.zeros([self.est_node_size, self.hid_feats], dtype=torch.float32, device=self.device))

        self.stream = torch.cuda.current_stream(self.device)
        self.shuffle = SageP3Shuffle.apply
        self.sampling_timers = []
        self.feature_timers = []
        self.backward_timers = []
        self.forward_timers = []
        self.duration = 0.0
        
    # fetch partial hid_feat from remote GPUs before forward pass
    # fetch partial gradient from remote GPUs during backward pass
    def _run_epoch(self, epoch):
        start = time.time()
        iter_idx = 0
        sampling_timer = CudaTimer()
        for input_nodes, output_nodes, blocks in self.train_data:
            sampling_timer.end()
            feat_timer = CudaTimer()
            top_block = blocks[0]
            iter_idx += 1
            # 1. Send and Receive edges for all the other gpus
            src, dst = top_block.adj_tensors('coo') # dgl v1.1 and above
            # src, dst = top_block.adj_sparse(fmt="coo") # dgl v1.0 and below
            self.edge_size_lst[self.rank] = (self.rank, src.shape[0], top_block.num_src_nodes(), top_block.num_dst_nodes()) # rank, edge_size, input_node_size
            dist.all_gather_object(object_list=self.edge_size_lst, obj=self.edge_size_lst[self.rank])
            # print(f"{self.rank=} {epoch=} {iter_idx=} {self.edge_size_lst=} start sending edges")
            for rank, edge_size, src_node_size, dst_node_size in self.edge_size_lst:
                self.src_edge_buffer_lst[rank].resize_(edge_size)
                self.dst_edge_buffer_lst[rank].resize_(edge_size)
                self.input_node_buffer_lst[rank].resize_(src_node_size)

            handle1 = dist.all_gather(tensor_list=self.input_node_buffer_lst, tensor=input_nodes, async_op=True)
            handle2 = dist.all_gather(tensor_list=self.src_edge_buffer_lst, tensor=src, async_op=True)
            handle3 = dist.all_gather(tensor_list=self.dst_edge_buffer_lst, tensor=dst, async_op=True)
            handle1.wait()
            for rank, _input_nodes in enumerate(self.input_node_buffer_lst):
                if "gpu" in self.config.system:
                    self.input_feat_buffer_lst[rank] = self.local_feat[_input_nodes]
                else:
                    self.input_feat_buffer_lst[rank] = gather_pinned_tensor_rows(self.local_feat, _input_nodes)
            handle2.wait()
            handle3.wait()
            feat_timer.end()
            # print(f"{self.rank=} {epoch=} {iter_idx=} input_feat_shapes={[x.shape for x in self.input_feat_buffer_lst]} start computing first hidden layer")
            forward_timer = CudaTimer()
            # 3. Fetch feature data and compute hid feature for other GPUs
            block = None
            for r in range(self.world_size):
                input_nodes = self.input_node_buffer_lst[r]
                input_feats = self.input_feat_buffer_lst[r]
                if r == self.rank:
                    block = top_block
                else:
                    src = self.src_edge_buffer_lst[r]
                    dst = self.dst_edge_buffer_lst[r]
                    src_node_size = self.edge_size_lst[r][2]
                    dst_node_size = self.edge_size_lst[r][3]
                    block = create_block(('coo', (src, dst)), num_dst_nodes=dst_node_size, num_src_nodes=src_node_size, device=self.device)
                                        
                self.local_hid_buffer_lst[r] = self.local_model(block, input_feats)
                self.global_grad_lst[r].resize_([block.num_dst_nodes(), self.hid_feats])

            # print(f"{self.rank=} {epoch=} {iter_idx=} start reduce first hidden layer features")
            # dist.barrier()
            local_hid: torch.Tensor = self.shuffle(self.rank, self.world_size, self.local_hid_buffer_lst[self.rank], self.local_hid_buffer_lst, self.global_grad_lst)
            output_labels = self.node_labels[output_nodes]
            
            # print(f"{self.rank=} {epoch=} {iter_idx=} local_hid_shape={[x.shape for x in self.local_hid_buffer_lst]} start compute remaining layer features")

            # 6. Compute forward pass locally
            output_pred = self.model(blocks[1:], local_hid)      
            forward_timer.end()      
            loss = F.cross_entropy(output_pred, output_labels) 
            backward_timer = CudaTimer()

            self.gloabl_optimizer.zero_grad()
            self.local_optimizer.zero_grad()
            loss.backward()
            self.gloabl_optimizer.step()
            # self.local_optimizer.step()
            # print(f"{self.rank=} {epoch=} {iter_idx=} global_grad_shape={[x.shape for x in self.global_grad_lst]} start gather error gradient")
            for r, global_grad in enumerate(self.global_grad_lst):
                if r != self.rank:
                    self.local_optimizer.zero_grad()
                    self.local_hid_buffer_lst[r].backward(global_grad)
                    self.local_optimizer.step()
            # print(f"{self.rank=} {epoch=} {iter_idx=} done")
            # self.local_optimizer.step()
            backward_timer.end()
            
            self.sampling_timers.append(sampling_timer)
            self.feature_timers.append(feat_timer)
            self.forward_timers.append(forward_timer)
            self.backward_timers.append(backward_timer)

            sampling_timer = CudaTimer()

        torch.cuda.synchronize()
        end = time.time()
        duration = end - start
        print(f"{self.rank=} {epoch=} {duration=}")
        # print(f"start evaluation for epoch {epoch}")
        # acc = self.evaluate()
    def log(self):
        sampling_time = get_duration(self.sampling_timers)
        feature_time = get_duration(self.feature_timers)
        forward_time = get_duration(self.forward_timers)
        backward_time = get_duration(self.backward_timers)
        profiler = Profiler(duration=self.duration, 
                            sampling_time=sampling_time, 
                            feature_time=feature_time, 
                            forward_time=forward_time, 
                            backward_time=backward_time, 
                            test_acc=self.test_acc)
        if self.rank == 0:
            write_to_csv( self.config.log_path, [self.config], [profiler])
            
    def train(self):
        self.model.train()
        self.local_model.train()
        timer = Timer()
        for epoch in range(self.config.num_epoch):
            self._run_epoch(epoch)
        self.duration = timer.duration()

        if self.rank == 0:
            print(f"train for {self.config.num_epoch} epochs in {self.duration}s")
        
    def evaluate(self):
        self.model.eval()
        self.local_model.eval()
        ys = []
        y_hats = []
        for it, (input_nodes, output_nodes, blocks) in enumerate(self.val_data):
            with torch.no_grad():
                top_block = blocks[0]
                # 1. Send and Receive edges for all the other gpus
                src, dst = top_block.adj_tensors('coo') # dgl v1.1 and above
                # src, dst = top_block.adj_sparse(fmt='coo') # dgl v1.0 and below
                self.edge_size_lst[self.rank] = (self.rank, src.shape[0], top_block.num_src_nodes(), top_block.num_dst_nodes()) # rank, edge_size, input_node_size
                dist.all_gather_object(object_list=self.edge_size_lst, obj=self.edge_size_lst[self.rank])
                for rank, edge_size, src_node_size, dst_node_size in self.edge_size_lst:
                    self.src_edge_buffer_lst[rank].resize_(edge_size)
                    self.dst_edge_buffer_lst[rank].resize_(edge_size)
                    self.input_node_buffer_lst[rank].resize_(src_node_size)
                    # self.input_feat_buffer_lst[rank].resize_([src_node_size, self.local_feat_width])
                # dist.barrier()
                handle1 = dist.all_gather(tensor_list=self.input_node_buffer_lst, tensor=input_nodes, async_op=True)
                handle2 = dist.all_gather(tensor_list=self.src_edge_buffer_lst, tensor=src, async_op=True)
                handle3 = dist.all_gather(tensor_list=self.dst_edge_buffer_lst, tensor=dst, async_op=True)
                handle1.wait()
                for rank, _input_nodes in enumerate(self.input_node_buffer_lst):
                    if "gpu" in self.config.system:
                        self.input_feat_buffer_lst[rank] = self.local_feat[_input_nodes]
                    else:
                        self.input_feat_buffer_lst[rank] = gather_pinned_tensor_rows(self.local_feat, _input_nodes)

                handle2.wait()
                handle3.wait()
                # 3. Fetch feature data and compute hid feature for other GPUs
                block = None
                for r in range(self.world_size):
                    input_nodes = self.input_node_buffer_lst[r]
                    input_feats = self.input_feat_buffer_lst[r]
                    if r == self.rank:
                        block = top_block
                    else:
                        src = self.src_edge_buffer_lst[r]
                        dst = self.dst_edge_buffer_lst[r]
                        src_node_size = self.edge_size_lst[r][2]
                        dst_node_size = self.edge_size_lst[r][3]
                        block = create_block(('coo', (src, dst)), num_dst_nodes=dst_node_size, num_src_nodes=src_node_size, device=self.device)
                                            
                    self.local_hid_buffer_lst[r] = self.local_model(block, input_feats)
                    # self.global_grad_lst[r].resize_([block.num_dst_nodes(), self.hid_feats])
                    # del block
                    
                local_hid = self.shuffle(self.rank, self.world_size, self.local_hid_buffer_lst[self.rank], self.local_hid_buffer_lst, None)
                ys.append(self.node_labels[output_nodes])
                y_hats.append(self.model(blocks[1:], local_hid))
                
        acc = MF.accuracy(
            torch.cat(y_hats),
            torch.cat(ys),
            task="multiclass",
            num_classes=self.num_classes)
    
        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        self.test_acc = (acc / self.world_size).item()
        return self.test_acc