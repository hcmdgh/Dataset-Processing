from gh import * 
from torch_geometric.datasets import DBLP 


def count_label(label, total):
    N = len(label)
    print(f"{N} ({int(N * 100 / total)}%)")
    
    counter = Counter(label.tolist())
    d = dict(counter)
    sum_ = sum(d.values())
    cnt_list = list(d.items())
    cnt_list.sort(key=lambda x: -x[1])
    
    str_list = []
    
    for lb, cnt in cnt_list:
        percent = int(cnt * 100 / sum_)
        str_list.append(f"{lb}: {cnt} ({percent}%)") 

    print(', '.join(str_list))


d = DBLP(root='/Dataset/PyG/DBLP/Raw')
hg = d.data 
print(hg)

label = hg['author'].y 
count_label(label, hg['author'].num_nodes)

train_mask = hg['author'].train_mask
val_mask = hg['author'].val_mask
test_mask = hg['author'].test_mask
count_label(label[train_mask], hg['author'].num_nodes)
count_label(label[val_mask], hg['author'].num_nodes)
count_label(label[test_mask], hg['author'].num_nodes)
