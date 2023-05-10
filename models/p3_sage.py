# Contruct a two-layer GNN model
from dgl.nn.pytorch.conv import SAGEConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from dgl import DGLGraph


def create_p3model(in_feats:int, hid_feats:int, num_classes:int, num_layers: int) -> tuple[nn.Module, nn.Module]:
    first_layer = SAGEConv(in_feats=in_feats, out_feats=hid_feats, aggregator_type="mean")
    remain_layers = P3_SAGE(in_feats, hid_feats, num_layers, num_classes)
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
        ctx.self_rank: int = self_rank
        ctx.world_size: int = world_size
        ctx.global_grads: list[torch.Tensor] = global_grads
        aggregated_hids = torch.clone(local_hid)
        for r in range(world_size):
            if r == self_rank:
                dist.reduce(tensor=aggregated_hids, dst=r, async_op=False) # gathering data from other GPUs
            else:
                dist.reduce(tensor=local_hids[r], dst=r, async_op=False) # TODO: Async gathering data from other GPUs
        return aggregated_hids
    
    @staticmethod
    def backward(ctx, grad_outputs):
        dist.all_gather(tensor=grad_outputs, tensor_list=ctx.global_grads)
        # print(f"backward {ctx.self_rank} {grad_outputs.shape=} {ctx.global_grads=}")
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