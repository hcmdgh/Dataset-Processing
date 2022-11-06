import numpy as np
from data import renamed_load

with open("/Dataset/OAG-from-HGT/Raw/graph_CS.pk", 'rb') as fp:
    graph = renamed_load(fp)
    
feat_paper = np.array(list(graph.node_feature['paper']['emb']))

np.save(
    file = '/Dataset/OAG-from-HGT/Processed/OAG-CS/feat_paper.npy', 
    arr = feat_paper,
)
