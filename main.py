
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Subset, DataLoader
from torch.utils.data import TensorDataset
import functions
import process_data
import random
from datetime import datetime
import random

## dataset-specific configs
## add a new entry here whenever you want to support another dataset

DATASET_CONFIGS = {
    "cifar10":  {"data_class": 10,  "lr": 1e-4, "weight_decay": 3e-4},
    "cifar100": {"data_class": 100, "lr": 1e-4, "weight_decay": 2e-4},
    "svhn":     {"data_class": 10,  "lr": 1e-4, "weight_decay": 4e-4},
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    default="cifar10",
    choices=DATASET_CONFIGS.keys(),
    help="which dataset to run (default: cifar10)",
)
parser.add_argument(
    "--num_client",
    type=int,
    default=20,
    help="number of clients to partition data across (default: 20)",
)
parser.add_argument(
    "--beta",
    type=float,
    default=0.01,
    help="Dirichlet concentration parameter, used only when --partition=dir; smaller = more skewed (default: 0.01)",
)
parser.add_argument(
    "--partition",
    default="dir",
    choices=["iid", "dir", "noniid"],
    help="how to partition data across clients: iid, dir, or noniid (default: dir)",
)
parser.add_argument(
    "--label_num_per_client",
    type=int,
    default=3,
    help="number of distinct classes per client, used only when --partition=noniid (default: 3)",
)
args = parser.parse_args()
dataset_name = args.dataset.lower()
cfg = DATASET_CONFIGS[dataset_name]
print(f"[config] dataset={dataset_name} | data_class={cfg['data_class']}| num_client={args.num_client} | partition={args.partition} | "
      f"beta={args.beta} | label_num_per_client={args.label_num_per_client}")

epochs=100
total_model=10

num_client=args.num_client
data_class=cfg["data_class"]
### first get data

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)),
])

if dataset_name == "svhn":
    train_ds = datasets.SVHN(root="./data", split='train', download=True, transform=transform)
    test_ds  = datasets.SVHN(root="./data", split='test',  download=True, transform=transform)
elif dataset_name == "cifar100":
    train_ds = datasets.CIFAR100(root="./data", train=True,  download=True, transform=transform)
    test_ds  = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
else:
    train_ds = datasets.CIFAR10(root="./data", train=True,  download=True, transform=transform)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)


train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### group data iid or non iid

net_dataidx_map = process_data.partition_data(train_loader=train_loader, partition=args.partition, num_users=num_client, beta=args.beta,label_num_per_client=args.label_num_per_client)
user_datasets = {}
user_loaders = {}
for user_id, idxs in net_dataidx_map.items():
    if idxs is None or len(idxs) == 0:  
        user_loaders[user_id] =None
        continue
    user_datasets[user_id] = Subset(train_loader.dataset, idxs)
    user_loaders[user_id] = DataLoader(
        user_datasets[user_id],
        batch_size=train_loader.batch_size,
        shuffle=True
    )

### create model for each client

clients = {}
kernel_sizes=[3,5,7]
strides=[1,2]

lists=np.random.choice(range(0,1000),size=total_model,replace=False)
for j in range(total_model):
    clients[j] = functions.RandomFeaturesCNN(feat_dim=256, kernel_size=random.choice(kernel_sizes),stride=random.choice(strides)).to(device).eval()   # frozen extractor, eval mode


head_classifier = functions.MLPHead(num_classes=data_class).to(device)
# optimizer = torch.optim.Adam(head_classifier.parameters(), lr=1e-4, weight_decay=3e-4) # 10
optimizer = torch.optim.Adam(head_classifier.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
criterion = nn.CrossEntropyLoss()

### train on all dataset

all_feats=[]
all_labels=[]


for j in range(num_client):    
    if user_loaders[j] is None:
        continue 
    for x, y in user_loaders[j]:
        x, y = x.to(device), y.to(device)
        temp_feats=[]
        for q in range(total_model): 
            client = clients[q].eval()
            with torch.no_grad():
                feats = client(x) 
                feats = feats.detach().cpu()
            temp_feats.append(feats)
        
        batch_feats = torch.cat(temp_feats, dim=1)
        # print('temp all feat',batch_feats.shape)
        all_feats.append(batch_feats)
        all_labels.append(y)
                
all_feats = torch.cat(all_feats, dim=0)     
all_labels = torch.cat(all_labels, dim=0)   
# print('all feats',all_feats.shape)

batch_size = 256
feat_dataset = TensorDataset(all_feats, all_labels)
feat_loader = DataLoader(feat_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

head_classifier.train()
for epoch in range(epochs):
    total_loss = 0.0
    correct = 0
    total = 0

    for feats, y in feat_loader:
        feats = feats.to(device)
        y = y.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        logits = head_classifier(feats)          
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),f"Epoch {epoch+1}/{epochs} | loss={total_loss/total:.4f}")

### eveluate on all test
def evaluate_test_global(test_loader, extractor, head, device,num_model):

    total_correct = 0
    total_samples = 0
    
    head.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            all_preds = []
            temp_list=[]
            for q in range(num_model):   
                extrat=  extractor[q].eval()           
                with torch.no_grad():
                    feats = extrat(x) 
                temp_list.append(feats)                  
            batch_feats = torch.cat(temp_list, dim=1)

            logits = head(batch_feats)
            probs = logits.argmax(dim=1) 
            all_preds.append(probs)
            
            all_preds = torch.stack(all_preds, dim=0)
            final_preds, _ = torch.mode(all_preds, dim=0)

            bs = x.size(0)
            total_correct += (final_preds == y).sum().item()
            total_samples += bs

    avg_acc  = total_correct / total_samples
    return avg_acc

test_acc=evaluate_test_global(test_loader, clients, head_classifier, device,total_model)
print("test accuracy",test_acc)