import os 
import sys 
os.chdir(os.path.dirname(__file__))
sys.path.append('../submodule/package')

from dgl_wrapper import * 
from jojo_es import * 

BATCH_SIZE = 1000 


def main():
    es_client = ESClient(
        host = '192.168.1.153', 
        port = 10000, 
        password = '6GYZTyH6D3fR4Y', 
    )
    paper_author_affiliation_index = es_client.get_index('mag_paper_author_affiliation')
    paper_index = es_client.get_index('mag_paper')

    dataset = homo_graph_dataset.OGB.OgbnArxiv(add_self_loop=False, to_undirected=False)
    g = dataset.g 
    print(g)
    
    with open('/home/gh/dataset/homo_graph/OGB/ogbn-arxiv/raw/ogbn_arxiv/mapping/nodeidx2paperid.csv', 'r', encoding='utf-8') as fp:
        reader = csv.DictReader(fp)
        entry_list = list(reader)

    num_nodes = len(entry_list)
    assert num_nodes == g.num_nodes 

    paper_magid_2_nid_map: dict[int, int] = dict()
    author_magid_2_nid_map: dict[int, int] = dict()
    institution_magid_2_nid_map: dict[int, int] = dict()
    PA_edge_list: list[tuple[int, int]] = list() 
    AI_edge_list: list[tuple[int, int]] = list() 
    
    for i in tqdm(range(0, num_nodes, BATCH_SIZE)):
        entry_batch = entry_list[i : i + BATCH_SIZE]

        paper_magid_set: set[int] = set() 
        
        for entry in entry_batch:
            paper_nid = int(entry['node idx'])
            paper_magid = int(entry['paper id'])

            paper_magid_2_nid_map[paper_magid] = paper_nid 
            paper_magid_set.add(paper_magid)
        
        doc_list = paper_author_affiliation_index.query_X_in_x('paper_id', paper_magid_set)
        assert len(doc_list) < 10000 
        
        for doc in doc_list:
            paper_magid = int(doc['paper_id'])
            paper_nid = paper_magid_2_nid_map[paper_magid]
            
            try:
                author_magid = int(doc['author_id']) 
            except Exception:
                author_magid = None
                
            try:
                institution_magid = int(doc['affiliation_id']) 
            except Exception:
                institution_magid = None
               
            if author_magid: 
                if author_magid not in author_magid_2_nid_map:
                    author_magid_2_nid_map[author_magid] = len(author_magid_2_nid_map) 

                author_nid = author_magid_2_nid_map[author_magid]
                PA_edge_list.append((paper_nid, author_nid))
                
                if institution_magid:
                    if institution_magid not in institution_magid_2_nid_map:
                        institution_magid_2_nid_map[institution_magid] = len(institution_magid_2_nid_map) 

                    institution_nid = institution_magid_2_nid_map[institution_magid]
                    AI_edge_list.append((author_nid, institution_nid))

    print(f"PA: {len(PA_edge_list)}")
    print(f"AI: {len(AI_edge_list)}")

    with open('/home/gh/dataset/hetero_graph/OGB/ogbn-arxiv/processed/extra_edge_index.dict.pkl', 'wb') as fp:
        extra_edge_index_dict = {
            ('paper', 'pa', 'author'): torch.tensor(PA_edge_list, dtype=torch.int64).T, 
            ('author', 'ai', 'institution'): torch.tensor(AI_edge_list, dtype=torch.int64).T, 
        }

        pickle.dump(extra_edge_index_dict, fp)
    
    
if __name__ == '__main__':
    main() 
