from ..imports import * 

__all__ = [
    'convert_edge_index_to_coo_matrix', 
]


def convert_edge_index_to_coo_matrix(edge_index: EdgeIndex, 
                                     num_nodes: Union[int, tuple[int, int]]) -> sp.coo_matrix: 
    assert edge_index.shape == (2, num_edges := edge_index.shape[-1])
                                     
    row, col = edge_index.detach().cpu().numpy() 
    ones = np.ones(num_edges) 
    
    if isinstance(num_nodes, (tuple, list)): 
        shape = [num_nodes[0], num_nodes[1]] 
    elif isinstance(num_nodes, int): 
        shape = [num_nodes] 
    else:
        raise TypeError 
        
    coo_matrix = sp.coo_matrix((ones, (row, col)), shape=shape) 
    
    return coo_matrix 
