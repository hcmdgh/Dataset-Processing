import pickle 
from collections import Counter 


def count_label(label, total):
    N = len(label)
    counter = Counter(label.tolist())
    d = dict(counter)
    
    print(f"{len(d)}类，{N} ({int(N * 100 / total)}%)")
    
    sum_ = sum(d.values())
    cnt_list = list(d.items())
    cnt_list.sort(key=lambda x: -x[1])
    
    str_list = []
    
    for lb, cnt in cnt_list:
        percent = int(cnt * 100 / sum_)
        str_list.append(f"{lb}: {cnt} ({percent}%)") 

    print(', '.join(str_list))


with open('/Dataset/PyG/ogbn-mag/Processed/ogbn-mag-TransE.dglhg.pkl', 'rb') as fp:
    hg = pickle.load(fp)
    
print(hg)
print(f"total: {hg.num_nodes()}, {hg.num_edges()}")
for ntype in hg.ntypes:
    print(f"{ntype} feat: {hg.nodes[ntype].data['feat'].shape}")

INFER_NTYPE = 'paper'

label = hg.nodes[INFER_NTYPE].data['label']
count_label(label, hg.num_nodes(INFER_NTYPE))

train_mask = hg.nodes[INFER_NTYPE].data['train_mask']
val_mask = hg.nodes[INFER_NTYPE].data['val_mask']
test_mask = hg.nodes[INFER_NTYPE].data['test_mask']
count_label(label[train_mask], hg.num_nodes(INFER_NTYPE))
count_label(label[val_mask], hg.num_nodes(INFER_NTYPE))
count_label(label[test_mask], hg.num_nodes(INFER_NTYPE))
