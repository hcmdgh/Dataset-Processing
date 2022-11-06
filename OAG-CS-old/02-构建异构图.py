import dgl 
import pickle 
import numpy as np 
import torch 
from collections import Counter


def main():
    feat_paper = np.load('/Dataset/OAG-from-HGT/Processed/OAG-CS/feat_paper.npy')
    feat_paper = torch.tensor(feat_paper, dtype=torch.float32)
    
    with open('/Dataset/OAG-from-HGT/Processed/OAG-CS/OAG-Venue.pkl', 'rb') as fp:
        graph_dict = pickle.load(fp)
        
    edge_index_dict = graph_dict['edges']
    label = graph_dict['labels']
    train_idx = graph_dict['split'][0]
    val_idx = graph_dict['split'][1]
    test_idx = graph_dict['split'][2]
    num_classes = graph_dict['n_classes'] 
    
    hg = dgl.heterograph(edge_index_dict)
    print(hg)    


if __name__ == '__main__':
    main() 
