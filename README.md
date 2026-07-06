# FedRFE
[UAI 2026] One-Shot Federated Learning based on Random Feature Extractor

## Usage

Run with default settings (CIFAR-10, 20 clients, Dirichlet partition, beta=0.01):
```
python main.py
```

Choose a different dataset:
```
python main.py --dataset svhn
python main.py --dataset cifar100
```

Change the number of clients:
```
python main.py --dataset cifar10 --num_client 5
```

Choose a data partitioning strategy (`iid`, `dir`, or `noniid`):
```
python main.py --partition iid
python main.py --partition dir --beta 0.05
python main.py --partition noniid --label_num_per_client 4
```

Combine any of the above:
```
python main.py --dataset svhn --num_client 20 --partition dir --beta 0.05
```

### Arguments

| Flag | Description | Default |
|---|---|---|
| `--dataset` | Dataset to use: `cifar10`, `cifar100`, or `svhn` | `cifar10` |
| `--num_client` | Number of clients to partition data across | `20` |
| `--partition` | Partitioning strategy: `iid`, `dir`, or `noniid` | `dir` |
| `--beta` | Dirichlet concentration parameter (used only with `--partition dir`); smaller = more skewed | `0.01` |
| `--label_num_per_client` | Number of classes per client (used only with `--partition noniid`) | `3` |
