import torch 
import dgl 
from collections import Counter
import pickle 
from ogb.nodeproppred import PygNodePropPredDataset


def count_label(label, total):
    N = len(label)
    print(f"{N} ({int(N * 100 / total)}%)")
    
    counter = Counter(label.tolist())
    d = dict(counter)
    
    print(f"num_classes: {len(d)}")
    
    sum_ = sum(d.values())
    cnt_list = list(d.items())
    cnt_list.sort(key=lambda x: -x[1])
    
    str_list = []
    
    for lb, cnt in cnt_list:
        percent = int(cnt * 100 / sum_)
        str_list.append(f"{lb}: {cnt} ({percent}%)") 

    print(', '.join(str_list))
    

dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='/Dataset/OGB/ogbn-arxiv/Raw') 

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"].squeeze(), split_idx["valid"].squeeze(), split_idx["test"].squeeze()

_g = dataset[0]
print(_g)

edge_index = tuple(_g.edge_index)
feat = _g.x 
label = _g.y.squeeze()
train_mask = torch.zeros(_g.num_nodes, dtype=torch.bool) 
val_mask = torch.zeros(_g.num_nodes, dtype=torch.bool) 
test_mask = torch.zeros(_g.num_nodes, dtype=torch.bool) 
train_mask[train_idx] = True
val_mask[valid_idx] = True
test_mask[test_idx] = True

count_label(label, _g.num_nodes)
count_label(label[train_mask], _g.num_nodes)
count_label(label[val_mask], _g.num_nodes)
count_label(label[test_mask], _g.num_nodes)

g = dgl.graph(edge_index, num_nodes=_g.num_nodes)
g.ndata['feat'] = feat 
g.ndata['label'] = label
g.ndata['train_mask'] = train_mask
g.ndata['val_mask'] = val_mask
g.ndata['test_mask'] = test_mask
print(g)

g = dgl.to_bidirected(g, copy_ndata=True)
g = dgl.add_self_loop(dgl.remove_self_loop(g))
print(g)

with open('/Dataset/OGB/ogbn-arxiv/Processed/ogbn-arxiv.dglg.pkl', 'wb') as fp:
    pickle.dump(g, fp)
