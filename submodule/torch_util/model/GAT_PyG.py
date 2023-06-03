from torch_geometric.nn import GATConv 

from ..imports import * 

__all__ = [
    'GAT_PyG', 
]


class GAT_PyG(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int, 
                 num_layers: int,
                 num_heads: int,
                 attn_dropout: float = 0., 
                 activation: nn.Module = nn.ELU(), 
                 dropout: float = 0.):
        super().__init__()

        self.num_layers = num_layers 
        
        self.conv_list = nn.ModuleList([
            GATConv(
                in_channels = in_dim, 
                out_channels = hidden_dim, 
                heads = num_heads, 
                concat = True, 
                dropout = attn_dropout, 
            ), 
            *[
                GATConv(
                    in_channels = hidden_dim * num_heads, 
                    out_channels = hidden_dim, 
                    heads = num_heads, 
                    concat = True, 
                    dropout = attn_dropout, 
                )
                for _ in range(num_layers - 2) 
            ],
            GATConv(
                in_channels = hidden_dim * num_heads, 
                out_channels = out_dim, 
                heads = num_heads, 
                concat = False,  
                dropout = attn_dropout, 
            ), 
        ])
        assert len(self.conv_list) == num_layers >= 2 
        
        self.activation = activation 
        
        self.dropout = nn.Dropout(dropout) 
        
    def forward(self,
                edge_index: EdgeIndex, 
                node_feat: FloatTensor) -> FloatTensor: 
        h = node_feat
                
        for l in range(self.num_layers): 
            h = self.conv_list[l](edge_index=edge_index, x=h) 
            
            if l < self.num_layers - 1: 
                h = self.activation(h) 
                h = self.dropout(h) 
                
        return h 
