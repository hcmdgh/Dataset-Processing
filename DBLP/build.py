from torch_geometric.datasets import DBLP 
import dgl 
import torch 
import pickle 

dataset = DBLP(root='/Dataset/PyG/DBLP/Raw')

_hg = dataset[0]
author_feat = _hg.x_dict['author']
author_label = _hg.y_dict['author']
author_train_mask = _hg.train_mask_dict['author']
author_val_mask = _hg.val_mask_dict['author']
author_test_mask = _hg.test_mask_dict['author']
paper_feat = _hg.x_dict['paper']
term_feat = _hg.x_dict['term']
edge_index_dict = {
    ('author', 'ap', 'paper'): tuple(_hg.edge_index_dict[('author', 'to', 'paper')]), 
    ('paper', 'pa', 'author'): tuple(_hg.edge_index_dict[('paper', 'to', 'author')]), 
    ('paper', 'pt', 'term'): tuple(_hg.edge_index_dict[('paper', 'to', 'term')]), 
    ('paper', 'pc', 'conference'): tuple(_hg.edge_index_dict[('paper', 'to', 'conference')]), 
    ('term', 'tp', 'paper'): tuple(_hg.edge_index_dict[('term', 'to', 'paper')]), 
    ('conference', 'cp', 'paper'): tuple(_hg.edge_index_dict[('conference', 'to', 'paper')]), 
}

hg = dgl.heterograph(edge_index_dict)
hg.nodes['author'].data['feat'] = author_feat
hg.nodes['author'].data['label'] = author_label
hg.nodes['author'].data['train_mask'] = author_train_mask
hg.nodes['author'].data['val_mask'] = author_val_mask
hg.nodes['author'].data['test_mask'] = author_test_mask
hg.nodes['paper'].data['feat'] = paper_feat
hg.nodes['term'].data['feat'] = term_feat

# Onehot
hg.nodes['conference'].data['feat'] = torch.eye(hg.num_nodes('conference'))
    
print(hg)
print(hg.num_nodes(), hg.num_edges())

with open('/Dataset/PyG/DBLP/Processed/DBLP.dglhg.pkl', 'wb') as fp:
    pickle.dump(hg, fp)
