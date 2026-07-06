
import numpy as np
import torch
from collections import defaultdict
import random
from itertools import combinations
from collections import defaultdict
import math

### group data iid


def size_of_division(num_groups, size):
    if isinstance(num_groups, int):
        num_per_group = [size // num_groups] * num_groups
    elif isinstance(num_groups, list):
        num_per_group = [math.floor(size * w) for w in num_groups]

    for i in np.random.choice(len(num_per_group), size - sum(num_per_group), replace=False):
        num_per_group[i] += 1
    
    return num_per_group


def generate_class_comb(num_groups, num_class, num_class_each_comb):
    class_comb = list(combinations(range(num_class), num_class_each_comb))
    np.random.shuffle(class_comb)
    if len(class_comb) <= num_groups:
        for _ in range(num_groups//len(class_comb)):
            class_comb += class_comb
    class_comb = class_comb[:num_groups]

    return class_comb

def partition_data(train_loader, partition, num_users, beta,label_num_per_client):
    n_parties = num_users
    data_size = len(train_loader.dataset)
    dataset=train_loader.dataset

    if hasattr(dataset, 'targets'):
        y_train = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        y_train = np.array(dataset.labels)
    else:
        raise AttributeError("Dataset has neither 'targets' nor 'labels' attribute")

    # Get num_classes regardless of dataset type
    if hasattr(dataset, 'classes'):
        num_classes = len(dataset.classes)
    else:
        num_classes = len(np.unique(y_train))

    # num_classes = len(train_loader.dataset.classes)
    # num_classes = len(np.unique(train_loader.dataset.labels))
    # y_train= np.array(train_loader.dataset.labels)
    # y_train= np.array(train_loader.dataset.targets)

    data_idxs_dict = {k: np.where(y_train == k)[0].tolist() for k in range(num_classes)}
    net_dataidx_map = defaultdict(list)
    label_num = len(data_idxs_dict)

    if partition == "iid":
        idxs = np.random.permutation(data_size)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}


    elif partition == "dir":
        
        # label -> indices
        # data_idxs_dict = {k: np.where(y_train == k)[0].tolist() for k in range(num_classes)}

        # net_dataidx_map = defaultdict(list)
    
        # label_num = len(data_idxs_dict)
        each_label_num = len(data_idxs_dict[0])
        
        scalenum = 10

        image_nums = []

        for n in range(label_num):
            image_num = []
            
            random.shuffle(data_idxs_dict[n])
            sampled_probabilities = int(scalenum) * each_label_num * np.random.dirichlet(
                np.array(num_users * [beta]))
            for user in range(num_users):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = data_idxs_dict[n][:min(len(data_idxs_dict[n]), no_imgs)]
                
                image_num.append(len(sampled_list))
                
                net_dataidx_map[user].extend(sampled_list)
                random.shuffle(data_idxs_dict[n])
            
            image_nums.append(image_num)
        
        net_dataidx_map =dict(net_dataidx_map)
    
    elif partition == "noniid":
        
        # client_idx_map = defaultdict(list)
        # label_num = len(data_idxs_dict)

        data_num_per_class = [len(data_idxs_dict[i]) for i in data_idxs_dict.keys()]

        datasize_per_client = size_of_division(num_users, sum(data_num_per_class))
        class_comb = generate_class_comb(num_users, label_num, label_num_per_client)
        
        datasize_per_client_per_class = [size_of_division(len(class_comb[i]), datasize_per_client[i]) for i in range(num_users)]

        temp = [set(i) for i in data_idxs_dict.values()]
        for c in range(num_users):
            cur_client_id = c
            for i in range(label_num_per_client):
                num_data_per_cli = datasize_per_client_per_class[c][i]
                cur_client_class_ = class_comb[c][i]

                if len(temp[cur_client_class_]) < num_data_per_cli:
                    rand_set = np.random.choice(list(temp[cur_client_class_]), num_data_per_cli, replace=True)
                elif len(temp[cur_client_class_]) == num_data_per_cli:
                    rand_set = np.random.choice(list(temp[cur_client_class_]), num_data_per_cli, replace=False)
                else:
                    rand_set = np.random.choice(list(temp[cur_client_class_]), num_data_per_cli, replace=False)
                    temp[cur_client_class_] = temp[cur_client_class_] - set(rand_set)  

                net_dataidx_map[cur_client_id].extend(rand_set)   


    return net_dataidx_map