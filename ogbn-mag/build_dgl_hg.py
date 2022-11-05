import dgl 
import torch 
import pickle 
from ogb.nodeproppred import DglNodePropPredDataset

dataset = DglNodePropPredDataset(name='ogbn-mag', root='/Dataset/OGB/ogbn-mag/Raw')

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"]['paper'].squeeze(), split_idx["valid"]['paper'].squeeze(), split_idx["test"]['paper'].squeeze()
graph, label = dataset[0]
print(graph)

edge_index_AI = graph.edges(etype='affiliated_with')
edge_index_AP = graph.edges(etype='writes')
edge_index_PP = graph.edges(etype='cites')
edge_index_PF = graph.edges(etype='has_topic')

hg = dgl.heterograph({
    ('author', 'AI', 'institution'): edge_index_AI,
    ('author', 'AP', 'paper'): edge_index_AP, 
    ('paper', 'PP_cites', 'paper'): edge_index_PP, 
    ('paper', 'PF', 'field'): edge_index_PF, 
    ('institution', 'IA', 'author'): edge_index_AI[::-1],
    ('paper', 'PA', 'author'): edge_index_AP[::-1], 
    ('paper', 'PP_cited', 'paper'): edge_index_PP[::-1], 
    ('field', 'FP', 'paper'): edge_index_PF[::-1], 
})

hg.nodes['paper'].data['year'] = graph.nodes['paper'].data['year'].squeeze()
hg.nodes['paper'].data['feat'] = graph.nodes['paper'].data['feat'].squeeze()
hg.nodes['paper'].data['label'] = label['paper'].squeeze()
train_mask = torch.zeros(hg.num_nodes('paper'), dtype=torch.bool)
val_mask = torch.zeros(hg.num_nodes('paper'), dtype=torch.bool)
test_mask = torch.zeros(hg.num_nodes('paper'), dtype=torch.bool)
train_mask[train_idx] = True 
val_mask[valid_idx] = True 
test_mask[test_idx] = True 
hg.nodes['paper'].data['train_mask'] = train_mask
hg.nodes['paper'].data['val_mask'] = val_mask
hg.nodes['paper'].data['test_mask'] = test_mask

print(hg)

with open('/Dataset/OGB/ogbn-mag/Processed/ogbn-mag.dglhg.pkl', 'wb') as fp:
    pickle.dump(hg, fp)
