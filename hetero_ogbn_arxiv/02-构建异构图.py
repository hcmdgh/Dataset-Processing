import os 
import sys 
os.chdir(os.path.dirname(__file__))
sys.path.append('../submodule/package')

from dgl_wrapper import * 


def main():
    dataset = homo_graph_dataset.OGB.OgbnArxiv(add_self_loop=False, to_undirected=False)
    g = dataset.g 
    edge_index = g.edge_index 
    label = dataset.label 
    num_papers = g.num_nodes
    paper_feat = dataset.feat
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    print(g)
    
    with open('/home/gh/dataset/hetero_graph/OGB/ogbn-arxiv/processed/extra_edge_index.dict.pkl', 'rb') as fp:
        edge_index_dict = pickle.load(fp)

    num_authors = int(torch.max(edge_index_dict[('paper', 'pa', 'author')][1])) + 1
    num_institutions = int(torch.max(edge_index_dict[('author', 'ai', 'institution')][1])) + 1 
    print(f"num_authors: {num_authors}")
    print(f"num_institutions: {num_institutions}")
        
    edge_index_dict[('author', 'ap', 'paper')] = torch.flip(edge_index_dict[('paper', 'pa', 'author')], dims=[0])
    edge_index_dict[('institution', 'ia', 'author')] = torch.flip(edge_index_dict[('author', 'ai', 'institution')], dims=[0])
    edge_index_dict[('paper', 'pp_cites', 'paper')] = edge_index 
    edge_index_dict[('paper', 'pp_cited', 'paper')] = torch.flip(edge_index, dims=[0]) 

    # 均值聚合，生成结点特征
    if True:
        bg = Graph.create_bipartite_graph(
            edge_index = edge_index_dict[('paper', 'pa', 'author')], 
            num_src_nodes = num_papers, 
            num_dest_nodes = num_authors, 
        )
        
        author_feat = generate_message_and_aggregate(
            g = bg, 
            message_func = MessageFunction.COPY_SRC, 
            reduce_func = ReduceFunction.MEAN,  
            src_feat = paper_feat, 
        )
        
        bg = Graph.create_bipartite_graph(
            edge_index = edge_index_dict[('author', 'ai', 'institution')], 
            num_src_nodes = num_authors, 
            num_dest_nodes = num_institutions, 
        )
        
        institution_feat = generate_message_and_aggregate(
            g = bg, 
            message_func = MessageFunction.COPY_SRC, 
            reduce_func = ReduceFunction.MEAN,  
            src_feat = author_feat, 
        )
        
    with open('/home/gh/dataset/hetero_graph/OGB/ogbn-arxiv/processed/hetero_ogbn_arxiv.dict.pkl', 'wb') as fp:
        graph_info = dict(
            edge_index_dict = edge_index_dict, 
            num_nodes_dict = dict(
                paper = num_papers, 
                author = num_authors, 
                institution = num_institutions,   
            ),
            infer_ntype = 'paper', 
            feat_dict = dict(
                paper = paper_feat, 
                author = author_feat, 
                institution = institution_feat,
            ),
            label = label, 
            train_mask = train_mask, 
            val_mask = val_mask, 
            test_mask = test_mask, 
        )
        
        pickle.dump(graph_info, fp)


if __name__ == '__main__':
    main() 
