import dgl.sparse as dglsp 

from ..imports import * 
from ..graph import convert_edge_index_to_coo_matrix, row_normalize, remove_diag 
from .etype import to_canonical_etype
from ..util import to_device 

__all__ = [
    'metapath_reachable_graph', 
    'propagate_metapath_feature', 
    'propagate_metapath_feature_without_self', 
    'compute_metapath_adjacency_matrix', 
]


def metapath_reachable_graph(edge_index_dict: dict[EdgeType, EdgeIndex],
                             num_nodes_dict: dict[NodeType, int],
                             metapath: list[str]) -> tuple[EdgeIndex, int, int]:
    device = next(iter(edge_index_dict.values())).device 
    
    A_accum = None 
                             
    for etype in metapath: 
        etype = to_canonical_etype(etype=etype, etypes=edge_index_dict)
        src_ntype, _, dest_ntype = etype 
        num_src_nodes, num_dest_nodes = num_nodes_dict[src_ntype], num_nodes_dict[dest_ntype] 
        edge_index = edge_index_dict[etype] 
        
        A = convert_edge_index_to_coo_matrix(
            edge_index = edge_index, 
            num_nodes = (num_src_nodes, num_dest_nodes), 
        ) 
        
        if A_accum is None: 
            A_accum = A 
        else:
            A_accum = A_accum @ A 
            
    assert A_accum is not None 

    num_src_nodes, num_dest_nodes = A_accum.shape 
    
    edge_index = np.stack(A_accum.nonzero()) 
    edge_index = torch.tensor(edge_index, dtype=torch.int64, device=device) 
     
    return edge_index, num_src_nodes, num_dest_nodes 


def propagate_metapath_feature(adj_mat_dict: dict[EdgeType, dglsp.SparseMatrix], 
                               metapath: list[str], 
                               src_feat: FloatTensor, 
                               use_gpu: bool = True) -> FloatTensor:
    if use_gpu:
        adj_mat_dict = to_device(adj_mat_dict) 
    else: 
        adj_mat_dict = to_device(adj_mat_dict, device='cpu') 
    
    aggr_feat = src_feat 
        
    for etype in metapath: 
        etype = to_canonical_etype(etype=etype, etypes=adj_mat_dict.keys()) 
        
        A = adj_mat_dict[etype]  
        A = A.transpose() 
        A = row_normalize(A) 
        
        aggr_feat = A @ aggr_feat  

    return aggr_feat  


def propagate_metapath_feature_without_self(adj_mat_dict: dict[EdgeType, dglsp.SparseMatrix],
                                            metapath: list[str], 
                                            src_feat: FloatTensor, 
                                            use_gpu: bool = True) -> FloatTensor:
    A_accum = compute_metapath_adjacency_matrix(
        adj_mat_dict = adj_mat_dict, 
        metapath = metapath, 
        use_gpu = use_gpu, 
    )
    
    A_accum = remove_diag(A_accum) 
    
    aggr_feat = A_accum @ src_feat 
        
    return aggr_feat 


def compute_metapath_adjacency_matrix(adj_mat_dict: dict[EdgeType, dglsp.SparseMatrix], 
                                      metapath: list[str], 
                                      use_gpu: bool = True) -> dglsp.SparseMatrix:
    if use_gpu:
        adj_mat_dict = to_device(adj_mat_dict) 
    else: 
        adj_mat_dict = to_device(adj_mat_dict, device='cpu') 
    
    A_prod = None 
        
    for etype in metapath: 
        etype = to_canonical_etype(etype=etype, etypes=adj_mat_dict.keys()) 
        
        A = adj_mat_dict[etype]  
        A = A.transpose() 
        A = row_normalize(A) 
        
        if A_prod is None: 
            A_prod = A 
        else: 
            A_prod = A @ A_prod  

    assert A_prod is not None 

    return A_prod 
