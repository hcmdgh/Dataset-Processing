from ..imports import * 

__all__ = [
    'edge_dropout', 
    'hetero_edge_dropout', 
    'node_feature_dropout', 
]


def edge_dropout(edge_index: EdgeIndex,
                 drop_ratio: float) -> tuple[EdgeIndex, BoolTensor]:
    device = edge_index.device 
    num_edges = edge_index.shape[-1]
    assert edge_index.shape == (2, num_edges) 
    
    drop_mask = torch.bernoulli(torch.full(fill_value=drop_ratio, size=[num_edges], device=device)).bool() 
    keep_mask = ~drop_mask 
    dropped_edge_index = edge_index[:, keep_mask]
    
    return dropped_edge_index, drop_mask 


def hetero_edge_dropout(edge_index_dict: dict[EdgeType, EdgeIndex],
                        drop_ratio: float) -> dict[EdgeType, EdgeIndex]: 
    dropped_edge_index_dict = {
        etype: edge_dropout(edge_index=edge_index, drop_ratio=drop_ratio)[0]
        for etype, edge_index in edge_index_dict.items() 
    }

    return dropped_edge_index_dict 


def node_feature_dropout(node_feat: FloatTensor,
                         drop_ratio: float, 
                         dim: int, 
                         fill_value: Any = 0.) -> tuple[FloatTensor, BoolTensor]: 
    device = node_feat.device 
    num_nodes, feat_dim = node_feat.shape 

    if dim == 0:
        drop_mask = torch.bernoulli(torch.full(fill_value=drop_ratio, size=[num_nodes], device=device)).bool() 
        dropped_feat = node_feat.clone() 
        dropped_feat[drop_mask, :] = fill_value 
        
        return dropped_feat, drop_mask 

    elif dim == 1 or dim == -1: 
        drop_mask = torch.bernoulli(torch.full(fill_value=drop_ratio, size=[feat_dim], device=device)).bool() 
        dropped_feat = node_feat.clone() 
        dropped_feat[:, drop_mask] = fill_value 
        
        return dropped_feat, drop_mask 
    
    else:
        raise AssertionError 
