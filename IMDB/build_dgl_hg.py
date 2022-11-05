from torch_geometric.datasets import IMDB 
import dgl 
import torch 
import pickle 

dataset = IMDB(root='/Dataset/PyG/IMDB/Raw')

_hg = dataset[0]
print(_hg)

feat = _hg.x_dict['movie']
label = _hg.y_dict['movie']
train_mask = _hg.train_mask_dict['movie']
val_mask = _hg.val_mask_dict['movie']
test_mask = _hg.test_mask_dict['movie']
edge_index_dict = {
    ('movie', 'MD', 'director'): tuple(_hg.edge_index_dict[('movie', 'to', 'director')]), 
    ('director', 'DM', 'movie'): tuple(_hg.edge_index_dict[('director', 'to', 'movie')]), 
    ('movie', 'MA', 'actor'): tuple(_hg.edge_index_dict[('movie', 'to', 'actor')]), 
    ('actor', 'AM', 'movie'): tuple(_hg.edge_index_dict[('actor', 'to', 'movie')]), 
}

hg = dgl.heterograph(edge_index_dict)
hg.nodes['movie'].data['feat'] = feat
hg.nodes['movie'].data['label'] = label
hg.nodes['movie'].data['train_mask'] = train_mask
hg.nodes['movie'].data['val_mask'] = val_mask
hg.nodes['movie'].data['test_mask'] = test_mask
hg.nodes['director'].data['feat'] = _hg.x_dict['director']
hg.nodes['actor'].data['feat'] = _hg.x_dict['actor']
print(hg)
print(hg.num_nodes(), hg.num_edges())

with open('/Dataset/PyG/IMDB/Processed/IMDB.dglhg.pkl', 'wb') as fp:
    pickle.dump(hg, fp)
