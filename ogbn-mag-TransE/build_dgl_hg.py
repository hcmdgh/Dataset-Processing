import dgl 
import torch 
import pickle 
from torch_geometric.datasets import OGB_MAG 

dataset = OGB_MAG(root='/Dataset/PyG/ogbn-mag/Raw', preprocess='TransE')

graph = dataset.data 
print(graph)

edge_index_AI = tuple(graph.edge_index_dict['author', 'affiliated_with', 'institution'])
edge_index_AP = tuple(graph.edge_index_dict['author', 'writes', 'paper'])
edge_index_PP = tuple(graph.edge_index_dict['paper', 'cites', 'paper'])
edge_index_PF = tuple(graph.edge_index_dict['paper', 'has_topic', 'field_of_study'])

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

hg.nodes['paper'].data['year'] = graph['paper'].year 
hg.nodes['paper'].data['feat'] = graph['paper'].x 
hg.nodes['paper'].data['label'] = graph['paper'].y 
hg.nodes['paper'].data['train_mask'] = graph['paper'].train_mask
hg.nodes['paper'].data['val_mask'] = graph['paper'].val_mask
hg.nodes['paper'].data['test_mask'] = graph['paper'].test_mask
hg.nodes['author'].data['feat'] = graph['author'].x 
hg.nodes['field'].data['feat'] = graph['field_of_study'].x 
hg.nodes['institution'].data['feat'] = graph['institution'].x 

print(hg)

with open('/Dataset/PyG/ogbn-mag/Processed/ogbn-mag-TransE.dglhg.pkl', 'wb') as fp:
    pickle.dump(hg, fp)
