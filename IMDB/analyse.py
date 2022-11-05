import pickle 
from collections import Counter 


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


with open('/Dataset/PyG/IMDB/Processed/IMDB.dglhg.pkl', 'rb') as fp:
    hg = pickle.load(fp)
    
print(hg)

label = hg.nodes['movie'].data['label']
count_label(label, hg.num_nodes('movie'))

train_mask = hg.nodes['movie'].data['train_mask']
val_mask = hg.nodes['movie'].data['val_mask']
test_mask = hg.nodes['movie'].data['test_mask']
count_label(label[train_mask], hg.num_nodes('movie'))
count_label(label[val_mask], hg.num_nodes('movie'))
count_label(label[test_mask], hg.num_nodes('movie'))
