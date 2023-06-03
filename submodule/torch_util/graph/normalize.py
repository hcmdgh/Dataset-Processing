from ..imports import * 

__all__ = [
    'normalize_node_feature', 
]


def normalize_node_feature(node_feat: FloatTensor) -> FloatTensor:
    num_nodes, feat_dim = node_feat.shape 
    
    out = node_feat - node_feat.min()
    out.div_(out.sum(dim=-1, keepdim=True).clamp_(min=1.))
    assert out.shape == (num_nodes, feat_dim) 
    
    return out 
