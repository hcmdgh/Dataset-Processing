from dgl.nn.pytorch import GraphConv as GCNConv 

from ..imports import * 

__all__ = [
    'GCN_DGL', 
]


class GCN_DGL(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 hidden_dim: int, 
                 out_dim: int, 
                 num_layers: int,
                 batch_norm: bool = False, 
                 activation: nn.Module = nn.ReLU(), 
                 dropout: float = 0.):
        super().__init__()

        self.num_layers = num_layers 
        
        self.conv_list = nn.ModuleList([
            GCNConv(
                in_feats = in_dim, 
                out_feats = hidden_dim,
            ), 
            *[
                GCNConv(
                    in_feats = hidden_dim, 
                    out_feats = hidden_dim,
                ) 
                for _ in range(num_layers - 2) 
            ],
            GCNConv(
                in_feats = hidden_dim, 
                out_feats = out_dim,
            ), 
        ])
        assert len(self.conv_list) == num_layers >= 2 
        
        if batch_norm: 
            self.bn_list = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) 
                for _ in range(num_layers - 1) 
            ])
        else:
            self.bn_list = None 
            
        self.activation = activation 
        
        self.dropout = nn.Dropout(dropout) 
        
    def reset_parameters(self): 
        for conv in self.conv_list: 
            conv.reset_parameters() 

    def forward(self,
                dgl_g: dgl.DGLGraph, 
                node_feat: FloatTensor) -> FloatTensor: 
        num_nodes, feat_dim = node_feat.shape  
        assert num_nodes == dgl_g.num_nodes()  
        
        h = node_feat
        
        for l in range(self.num_layers): 
            h = self.conv_list[l](graph=dgl_g, feat=h)          
            
            if l < self.num_layers - 1: 
                if self.bn_list is not None: 
                    h = self.bn_list[l](h) 
                h = self.activation(h) 
                h = self.dropout(h)
                            
        return h 
