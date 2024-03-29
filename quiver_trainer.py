import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.cuda import Event as event
import time
import csv
import torchmetrics.functional as MF
from utils import RunConfig, TrainProfiler, QuiverDglSageSample
import quiver

class QuiverTrainer:
    def __init__(
        self,
        config: RunConfig,
        model: torch.nn.Module,
        train_data: QuiverDglSageSample,
        val_data: QuiverDglSageSample,
        global_feat: quiver.Feature, # quiver Feature
        label: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        nid_dtype: torch.dtype = torch.int64
    ) -> None:
        self.config = config
        self.rank = config.rank
        self.world_size = config.world_size
        self.device = torch.device(f"cuda:{self.rank}")
        self.feat = global_feat
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
    
    def _run_epoch(self, epoch): 
        forward = 0.0
        backward = 0.0
        sample_time = 0.0
        feat_time = 0.0
        start = time.time()
        sample_start = time.time()
        iter_idx = 0
        for input_nodes, output_nodes, blocks in self.train_data:
            iter_idx = iter_idx + 1
            torch.cuda.synchronize(self.device)
            feat_start = sample_end = time.time()
            input_feats = self.feat[input_nodes.long()]
            output_labels = self.node_labels[output_nodes.long()]

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
        acc = self.evaluate()
        if self.rank == 0 or self.world_size == 1:
            other = epoch_time - forward - backward - feat_time - sample_time

            # info = self.log.log_step(epoch, acc, epoch_time, forward, backward, feat_time, sample_time)
            # print(info)
            
            fwd_l1_time = 0.0
            if self.world_size == 1:
                fwd_l1_time = self.model.fwd_l1_time() / 1000
            else:
                fwd_l1_time = self.model.module.fwd_l1_time() / 1000
                
            item = {
                "epoch": epoch,
                "val_acc": acc,
                "epoch_time": epoch_time,
                "forward": forward,
                "forward_top": fwd_l1_time,
                "forward_top_ratio": round(fwd_l1_time / forward, 5),
                "backward": backward,
                "feat": feat_time,
                "sample": sample_time,
                "other": other
            }    
            self.log.log_step_dict(item)
            print(item)
                        
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
        # print(f"eval {self.rank=}")
        self.model.eval()
        ys = []
        y_hats = []
        for it, (input_nodes, output_nodes, blocks) in enumerate(self.val_data):
            with torch.no_grad():
                x = self.feat[input_nodes.long()]
                ys.append(self.node_labels[output_nodes.long()])
                y_hats.append(self.model(blocks, x))
    
        acc = MF.accuracy(
            torch.cat(y_hats),
            torch.cat(ys),
            task="multiclass",
            num_classes=self.num_classes)
        
        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        return (acc / self.world_size).item()
    