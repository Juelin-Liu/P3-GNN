# P3-GNN
Implementation for "P3: Distributed Deep Graph Learning at Scale"

# The available configurations can be obtained from
```python
python3 run.py -h
```
For example to run `p3` on graph ogbn-products:

```python
python3 run.py --mode 3 --graph_name=ogbn-products
```

If your system doesn't have NVLINK, you might need to disable NCCL's P2P setting.
```python
NCCL_P2P_DISABLE=1 python3 run.py --mode 3 --graph_name=ogbn-products
```

The app uses a default batch size of 1024 and fanouts [20, 20, 20], which can be changed by modifiying the value in `run.py`.

# Dataset
The dataset will be downloaded into the `dataset` directory

# Output
The profiling data will be stored in the `logs` directory