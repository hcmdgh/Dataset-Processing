from torch_geometric.utils import coalesce 

from ..imports import * 

__all__ = [
    'to_undirected', 
    'add_reverse_edges', 
]


def to_undirected(edge_index: EdgeIndex, 
                  num_nodes: int) -> EdgeIndex:
    """
    将图转换为无向图，且每两个结点之间之多有一条边。 
    """
    
    assert edge_index.shape == (2, num_edges := edge_index.shape[-1]) 
    
    row, col = edge_index 
    row, col = torch.cat([row, col]), torch.cat([col, row])
    new_edge_index = torch.stack([row, col]) 

    return coalesce(
        edge_index = new_edge_index, 
        edge_attr = None, 
        num_nodes = num_nodes,
    )[0]


def add_reverse_edges(edge_index: EdgeIndex) -> EdgeIndex: 
    assert edge_index.shape == (2, num_edges := edge_index.shape[-1]) 
    
    row, col = edge_index 
    row, col = torch.cat([row, col]), torch.cat([col, row])
    new_edge_index = torch.stack([row, col]) 
    
    return new_edge_index 
