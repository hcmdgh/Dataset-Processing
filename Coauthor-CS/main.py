from torch_geometric.datasets import Coauthor
import dgl 
import torch 
import pickle 
from collections import Counter


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
    
    
dataset = Coauthor(
    root = '/Dataset/PyG/Coauthor/Raw',
    name = 'CS',
)

_g = dataset.data 
print(_g)

edge_index = tuple(_g.edge_index)
feat = _g.x
label = _g.y 

count_label(label=label, total=_g.num_nodes)

g = dgl.graph(edge_index, num_nodes=_g.num_nodes)
g.ndata['feat'] = feat 
g.ndata['label'] = label 

print(g)
g = dgl.to_bidirected(g, copy_ndata=True)
g = dgl.add_self_loop(dgl.remove_self_loop(g))
print(g)

with open('/Dataset/PyG/Coauthor/Processed/Coauthor-CS.dglg.pkl', 'wb') as fp:
    pickle.dump(g, fp)
