import torch 
import dgl 
import pickle 
import os 
import numpy as np 

# 将步骤00生成的相应文件放置在该目录下
ROOT = '/Dataset/OAG-from-HGT/Processed/OAG-CS'


def load_OAG_Venue_hg() -> dgl.DGLHeteroGraph:
    with open(os.path.join(ROOT, 'OAG-Venue/OAG-Venue.pkl'), 'rb') as fp:
        graph_dict = pickle.load(fp)

    _edge_index_dict = graph_dict["edges"] 
    edge_index_dict = {
        ('author', 'AP_first', 'paper'): _edge_index_dict[('author', 'AP_write_first', 'paper')], 
        ('author', 'AP_other', 'paper'): _edge_index_dict[('author', 'AP_write_other', 'paper')], 
        ('author', 'AP_last', 'paper'): _edge_index_dict[('author', 'AP_write_last', 'paper')], 
        ('author', 'AI', 'institution'): _edge_index_dict[('author', 'in', 'affiliation')], 
        ('field', 'FF', 'field'): _edge_index_dict[('field', 'FF_in', 'field')], 
        ('paper', 'PF_L0', 'field'): _edge_index_dict[('paper', 'PF_in_L0', 'field')], 
        ('paper', 'PF_L1', 'field'): _edge_index_dict[('paper', 'PF_in_L1', 'field')], 
        ('paper', 'PF_L2', 'field'): _edge_index_dict[('paper', 'PF_in_L2', 'field')], 
        ('paper', 'PF_L3', 'field'): _edge_index_dict[('paper', 'PF_in_L3', 'field')], 
        ('paper', 'PF_L4', 'field'): _edge_index_dict[('paper', 'PF_in_L4', 'field')], 
        ('paper', 'PF_L5', 'field'): _edge_index_dict[('paper', 'PF_in_L5', 'field')], 
        ('paper', 'PP_cite', 'paper'): _edge_index_dict[('paper', 'PP_cite', 'paper')], 
        ('paper', 'PV_conference', 'venue'): _edge_index_dict[('paper', 'PV_Conference', 'venue')], 
        ('paper', 'PV_patent', 'venue'): _edge_index_dict[('paper', 'PV_Patent', 'venue')], 
        ('paper', 'PV_repository', 'venue'): _edge_index_dict[('paper', 'PV_Repository', 'venue')], 
    }
    edge_index_dict[('paper', 'PA_first', 'author')] = edge_index_dict[('author', 'AP_first', 'paper')][::-1]
    edge_index_dict[('paper', 'PA_other', 'author')] = edge_index_dict[('author', 'AP_other', 'paper')][::-1]
    edge_index_dict[('paper', 'PA_last', 'author')] = edge_index_dict[('author', 'AP_last', 'paper')][::-1]
    edge_index_dict[('institution', 'IA', 'author')] = edge_index_dict[('author', 'AI', 'institution')][::-1]
    edge_index_dict[('field', 'FF_rev', 'field')] = edge_index_dict[('field', 'FF', 'field')][::-1]
    edge_index_dict[('field', 'FP_L0', 'paper')] = edge_index_dict[('paper', 'PF_L0', 'field')][::-1]
    edge_index_dict[('field', 'FP_L1', 'paper')] = edge_index_dict[('paper', 'PF_L1', 'field')][::-1]
    edge_index_dict[('field', 'FP_L2', 'paper')] = edge_index_dict[('paper', 'PF_L2', 'field')][::-1]
    edge_index_dict[('field', 'FP_L3', 'paper')] = edge_index_dict[('paper', 'PF_L3', 'field')][::-1]
    edge_index_dict[('field', 'FP_L4', 'paper')] = edge_index_dict[('paper', 'PF_L4', 'field')][::-1]
    edge_index_dict[('field', 'FP_L5', 'paper')] = edge_index_dict[('paper', 'PF_L5', 'field')][::-1]
    edge_index_dict[('paper', 'PP_cited', 'paper')] = edge_index_dict[('paper', 'PP_cite', 'paper')][::-1]
    edge_index_dict[('venue', 'VP_conference', 'paper')] = edge_index_dict[('paper', 'PV_conference', 'venue')][::-1]
    edge_index_dict[('venue', 'VP_patent', 'paper')] = edge_index_dict[('paper', 'PV_patent', 'venue')][::-1]
    edge_index_dict[('venue', 'VP_repository', 'paper')] = edge_index_dict[('paper', 'PV_repository', 'venue')][::-1]
    
    hg = dgl.heterograph(edge_index_dict)
    
    train_idx, val_idx, test_idx = graph_dict["split"]

    author_emb = torch.load(os.path.join(ROOT, 'OAG-Venue/author.pt')).float()
    field_emb = torch.load(os.path.join(ROOT, 'OAG-Venue/field.pt')).float()
    venue_emb = torch.load(os.path.join(ROOT, 'OAG-Venue/venue.pt')).float()
    affiliation_emb = torch.load(os.path.join(ROOT, 'OAG-Venue/affiliation.pt')).float()
    paper_feat = torch.from_numpy(np.load(os.path.join(ROOT, 'OAG-Venue/paper.npy'))).float()
    
    hg.nodes["paper"].data["feat"] = paper_feat[:hg.number_of_nodes("paper")]
    hg.nodes["author"].data["feat"] = author_emb[:hg.number_of_nodes("author")]
    hg.nodes["institution"].data["feat"] = affiliation_emb[:hg.number_of_nodes("institution")]
    hg.nodes["field"].data["feat"] = field_emb[:hg.number_of_nodes("field")]
    hg.nodes["venue"].data["feat"] = venue_emb[:hg.number_of_nodes("venue")]

    label = torch.from_numpy(graph_dict["labels"])
    hg.nodes['paper'].data['label'] = label 
    
    train_mask = torch.zeros(hg.num_nodes('paper'), dtype=torch.bool)
    val_mask = torch.zeros(hg.num_nodes('paper'), dtype=torch.bool)
    test_mask = torch.zeros(hg.num_nodes('paper'), dtype=torch.bool)
    train_mask[train_idx] = True 
    val_mask[val_idx] = True 
    test_mask[test_idx] = True 
    hg.nodes['paper'].data['train_mask'] = train_mask
    hg.nodes['paper'].data['val_mask'] = val_mask
    hg.nodes['paper'].data['test_mask'] = test_mask
    
    return hg 
    
        
def load_OAG_L1_Field_hg() -> dgl.DGLHeteroGraph:
    with open(os.path.join(ROOT, 'OAG-L1-Field/OAG-L1-Field.pkl'), 'rb') as fp:
        graph_dict = pickle.load(fp)

    _edge_index_dict = graph_dict["edges"] 
    edge_index_dict = {
        ('author', 'AP_first', 'paper'): _edge_index_dict[('author', 'AP_write_first', 'paper')], 
        ('author', 'AP_other', 'paper'): _edge_index_dict[('author', 'AP_write_other', 'paper')], 
        ('author', 'AP_last', 'paper'): _edge_index_dict[('author', 'AP_write_last', 'paper')], 
        ('author', 'AI', 'institution'): _edge_index_dict[('author', 'in', 'affiliation')], 
        ('field', 'FF', 'field'): _edge_index_dict[('field', 'FF_in', 'field')], 
        ('paper', 'PF_L0', 'field'): _edge_index_dict[('paper', 'PF_in_L0', 'field')], 
        ('paper', 'PF_L1', 'field'): _edge_index_dict[('paper', 'PF_in_L1', 'field')], 
        ('paper', 'PF_L2', 'field'): _edge_index_dict[('paper', 'PF_in_L2', 'field')], 
        ('paper', 'PF_L3', 'field'): _edge_index_dict[('paper', 'PF_in_L3', 'field')], 
        ('paper', 'PF_L4', 'field'): _edge_index_dict[('paper', 'PF_in_L4', 'field')], 
        ('paper', 'PF_L5', 'field'): _edge_index_dict[('paper', 'PF_in_L5', 'field')], 
        ('paper', 'PP_cite', 'paper'): _edge_index_dict[('paper', 'PP_cite', 'paper')], 
        ('paper', 'PV_conference', 'venue'): _edge_index_dict[('paper', 'PV_conference', 'venue')], 
        ('paper', 'PV_patent', 'venue'): _edge_index_dict[('paper', 'PV_Patent', 'venue')], 
        ('paper', 'PV_repository', 'venue'): _edge_index_dict[('paper', 'PV_Repository', 'venue')], 
    }
    edge_index_dict[('paper', 'PA_first', 'author')] = edge_index_dict[('author', 'AP_first', 'paper')][::-1]
    edge_index_dict[('paper', 'PA_other', 'author')] = edge_index_dict[('author', 'AP_other', 'paper')][::-1]
    edge_index_dict[('paper', 'PA_last', 'author')] = edge_index_dict[('author', 'AP_last', 'paper')][::-1]
    edge_index_dict[('institution', 'IA', 'author')] = edge_index_dict[('author', 'AI', 'institution')][::-1]
    edge_index_dict[('field', 'FF_rev', 'field')] = edge_index_dict[('field', 'FF', 'field')][::-1]
    edge_index_dict[('field', 'FP_L0', 'paper')] = edge_index_dict[('paper', 'PF_L0', 'field')][::-1]
    edge_index_dict[('field', 'FP_L1', 'paper')] = edge_index_dict[('paper', 'PF_L1', 'field')][::-1]
    edge_index_dict[('field', 'FP_L2', 'paper')] = edge_index_dict[('paper', 'PF_L2', 'field')][::-1]
    edge_index_dict[('field', 'FP_L3', 'paper')] = edge_index_dict[('paper', 'PF_L3', 'field')][::-1]
    edge_index_dict[('field', 'FP_L4', 'paper')] = edge_index_dict[('paper', 'PF_L4', 'field')][::-1]
    edge_index_dict[('field', 'FP_L5', 'paper')] = edge_index_dict[('paper', 'PF_L5', 'field')][::-1]
    edge_index_dict[('paper', 'PP_cited', 'paper')] = edge_index_dict[('paper', 'PP_cite', 'paper')][::-1]
    edge_index_dict[('venue', 'VP_conference', 'paper')] = edge_index_dict[('paper', 'PV_conference', 'venue')][::-1]
    edge_index_dict[('venue', 'VP_patent', 'paper')] = edge_index_dict[('paper', 'PV_patent', 'venue')][::-1]
    edge_index_dict[('venue', 'VP_repository', 'paper')] = edge_index_dict[('paper', 'PV_repository', 'venue')][::-1]
    
    hg = dgl.heterograph(edge_index_dict)
    train_idx, val_idx, test_idx = graph_dict["split"]

    author_emb = torch.load(os.path.join(ROOT, 'OAG-L1-Field/author.pt')).float()
    field_emb = torch.load(os.path.join(ROOT, 'OAG-L1-Field/field.pt')).float()
    venue_emb = torch.load(os.path.join(ROOT, 'OAG-L1-Field/venue.pt')).float()
    affiliation_emb = torch.load(os.path.join(ROOT, 'OAG-L1-Field/affiliation.pt')).float()
    paper_feat = torch.from_numpy(np.load(os.path.join(ROOT, 'OAG-L1-Field/paper.npy'))).float()
    
    hg.nodes["paper"].data["feat"] = paper_feat[:hg.number_of_nodes("paper")]
    hg.nodes["author"].data["feat"] = author_emb[:hg.number_of_nodes("author")]
    hg.nodes["institution"].data["feat"] = affiliation_emb[:hg.number_of_nodes("affiliation")]
    hg.nodes["field"].data["feat"] = field_emb[:hg.number_of_nodes("field")]
    hg.nodes["venue"].data["feat"] = venue_emb[:hg.number_of_nodes("venue")]

    num_classes = graph_dict["n_classes"]
    label = torch.zeros(hg.num_nodes('paper'), num_classes)
    for key in graph_dict["labels"]:
        label[key, graph_dict["labels"][key]] = 1
    hg.nodes['paper'].data['label'] = label 
    
    train_mask = torch.zeros(hg.num_nodes('paper'), dtype=torch.bool)
    val_mask = torch.zeros(hg.num_nodes('paper'), dtype=torch.bool)
    test_mask = torch.zeros(hg.num_nodes('paper'), dtype=torch.bool)
    train_mask[train_idx] = True 
    val_mask[val_idx] = True 
    test_mask[test_idx] = True 
    hg.nodes['paper'].data['train_mask'] = train_mask
    hg.nodes['paper'].data['val_mask'] = val_mask
    hg.nodes['paper'].data['test_mask'] = test_mask
    
    return hg 


def main():
    hg = load_OAG_Venue_hg()
    print(hg)
    
    with open(os.path.join(ROOT, 'OAG-Venue/OAG-Venue.dglhg.pkl'), 'wb') as fp:
        pickle.dump(hg, fp)
    
    # hg = load_OAG_L1_Field_hg()
    # print(hg)
        

if __name__ == '__main__':
    main() 
