# Contruct a two-layer GNN model
from dgl.nn.pytorch.conv import SAGEConv
import torch.nn as nn
import torch
import torch.distributed as dist

class SAGE(nn.Module):
    def __init__(self, in_feats: int, hid_feats: int, num_layers: int, out_feats: int):
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layers = nn.ModuleList()
        self.fwd_l1_timer = []    
        self.hid_feats_lst = []
        
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                self.layers.append(SAGEConv(in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean'))
            elif layer_idx >= 1 and layer_idx < num_layers - 1:            
                self.layers.append(SAGEConv(
                    in_feats=hid_feats, out_feats=hid_feats, aggregator_type='mean'))
            else:
                # last layer
                self.layers.append(SAGEConv(
                    in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean'))

    def forward(self, blocks, feat):
        hid_feats = feat
        l1_start = torch.cuda.Event(enable_timing=True)
        l1_start.record()
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            hid_feats = layer(block, hid_feats)
            if (layer_idx == 0):
                l1_end = torch.cuda.Event(enable_timing=True)
                l1_end.record()
                self.fwd_l1_timer.append((l1_start, l1_end))   
            if layer_idx != len(self.layers) - 1:
                hid_feats = self.activation(hid_feats)
                hid_feats = self.dropout(hid_feats)
        return hid_feats
    
    def fwd_l1_time(self):
        torch.cuda.synchronize()
        fwd_time = 0.0
        for l1_start, l1_end in self.fwd_l1_timer:
            fwd_time += l1_start.elapsed_time(l1_end)
        self.fwd_l1_timer = []
        return fwd_time
    
    
def create_p3model(rank:int, in_feats:int, hid_feats:int, num_classes:int, num_layers: int) -> tuple[nn.Module, nn.Module]:
    first_layer = SAGEConv(in_feats=in_feats, out_feats=hid_feats, aggregator_type="mean").to(rank) # Intra-Model Parallel
    remain_layers = P3_SAGE(in_feats, hid_feats, num_layers, num_classes).to(rank) # Data Parallel
    return (first_layer, remain_layers)


class ShuffleLayer(torch.autograd.Function):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @staticmethod
    def forward(ctx, 
                self_rank: int, 
                world_size:int,
                local_hid: torch.Tensor,
                local_hids: list[torch.Tensor],
                global_grads: list[torch.Tensor])->torch.Tensor:
        # print(f"forward {self_rank=} {world_size=} {local_hid.shape}")        
        ctx.self_rank = self_rank
        ctx.world_size = world_size
        ctx.global_grads = global_grads
        # aggregated_hid = torch.clone(local_hid)
        aggregated_hid = local_hid.detach().clone()
        handle = None
        for r in range(world_size):
            if r == self_rank:
                handle = dist.reduce(tensor=aggregated_hid, dst=r, async_op=True) # gathering data from other GPUs
            else:
                dist.reduce(tensor=local_hids[r], dst=r, async_op=True) # TODO: Async gathering data from other GPUs
        handle.wait()
        return aggregated_hid
    
    @staticmethod
    def backward(ctx, grad_outputs):
        # print(f"self.rank={ctx.self_rank} send_grad_shape={grad_outputs.shape} global_grads_shape={[x.shape for x in ctx.global_grads]}")
        dist.all_gather(tensor=grad_outputs, tensor_list=ctx.global_grads)
        return None, None, grad_outputs, None, None
    
    
class P3_SAGE(nn.Module):
    def __init__(self, 
                 in_feats: int,
                 hid_feats: int, 
                 num_layers: int, 
                 out_feats: int):
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                # first layer
                continue
                # self.layers.append(P3_SAGEConv(in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean'))
            elif layer_idx >= 1 and layer_idx < num_layers - 1:          
                # middle layers  
                self.layers.append(SAGEConv(
                    in_feats=hid_feats, out_feats=hid_feats, aggregator_type='mean'))
            else:
                # last layer
                self.layers.append(SAGEConv(
                    in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean'))

    def forward(self, blocks, feat):
        hid_feats = feat
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            hid_feats = layer(block, hid_feats)
            if layer_idx != len(self.layers) - 1:
                hid_feats = self.activation(hid_feats)
                hid_feats = self.dropout(hid_feats)
        return hid_feats