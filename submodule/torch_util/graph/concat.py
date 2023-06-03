from torch_geometric.utils import coalesce 

from ..imports import * 

__all__ = [
    'concat_edge_index', 
]


def concat_edge_index(*edge_index_list,
                      num_nodes: int) -> EdgeIndex:
    edge_index = torch.concat(edge_index_list, dim=-1) 
    assert edge_index.shape == (2, num_edges := edge_index.shape[-1]) 
    
    edge_index = coalesce(
        edge_index = edge_index, 
        edge_attr = None, 
        num_nodes = num_nodes,
    )[0]
    
    return edge_index 
