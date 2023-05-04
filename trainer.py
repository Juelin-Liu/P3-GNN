import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from dgl.dataloading import DataLoader as DglDataLoader
import time
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
        print("AVERAGE EPOCH TIME: ", self.avg_epoch())
        with open(self.path, "w+") as file:
            writer = csv.DictWriter(file, self.fields)
            writer.writeheader()
            for item in self.items:
                writer.writerow(item)

class Trainer:
    def __init__(
        self,
        config: RunConfig,
        model: torch.nn.Module,
        train_data: DglDataLoader,
        val_data: DglDataLoader,
        feat: torch.Tensor,
        label: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.config = config
        self.rank = config.rank
        self.world_size = config.world_size
        self.device = torch.device(f"cuda:{self.rank}")
        self.model = model.to(self.device)
        self.feat = feat.to(self.device)
        self.node_labels = torch.flatten(label).to(self.device)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.model = None
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
        for input_nodes, output_nodes, blocks in self.train_data:
            feat_start = sample_end = time.time()
            input_feats = self.feat[input_nodes]
            output_labels = self.node_labels[output_nodes]
            feat_end = forward_start = time.time()
            output_pred = self.model(blocks, input_feats)            
            loss = F.cross_entropy(output_pred, output_labels)
            forward_end = backward_start = time.time()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            backward_end = time.time()
            
            forward += forward_end - forward_start
            backward += backward_end - backward_start
            sample_time += sample_end - sample_start
            feat_time += feat_end - feat_start
            
            sample_start = time.time()
        end = time.time()
        epoch_time = end - start
        
        if self.rank == 0 or self.world_size == 1:
            acc = self.evaluate()
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
                x = self.feat[input_nodes]
                ys.append(self.node_labels[output_nodes])
                y_hats.append(self.model(blocks, x))
        return MF.accuracy(
            torch.cat(y_hats),
            torch.cat(ys),
            task="multiclass",
            num_classes=self.num_classes).item()
    