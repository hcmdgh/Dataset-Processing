from torch_geometric.loader import NeighborLoader as _NeighborLoader 
from torch_geometric.data import Data 

from ..imports import * 

__all__ = [
    'NeighborLoader', 
]


@dataclass
class GraphBlock:
    edge_index: EdgeIndex 
    num_nodes: int 
    num_target_nodes: int 
    target_node_mask: BoolTensor  
    target_node_idx: IntTensor 
    node_idx: IntTensor 
    edge_idx: IntTensor 


class NeighborLoader:
    def __init__(self,
                 edge_index: EdgeIndex, 
                 num_nodes: int,
                 target_node_idx: Optional[Tensor], 
                 num_neighbors: list[int],
                 batch_size: int,
                 shuffle: bool, 
                 drop_last: bool = False, 
                 **kwargs):
        self.edge_index = edge_index 
        self.num_nodes = num_nodes 
        self.num_edges = edge_index.shape[-1] 
        assert edge_index.shape == (2, self.num_edges) 
        self.batch_size = batch_size 
        
        data = Data(edge_index=edge_index, num_nodes=num_nodes) 
                 
        self.loader = _NeighborLoader(
            data = data, 
            num_neighbors = num_neighbors, 
            input_nodes = target_node_idx, 
            batch_size = batch_size, 
            shuffle = shuffle, 
            drop_last = drop_last, 
            **kwargs, 
        )
        
    def __len__(self) -> int:
        return len(self.loader) 
    
    def __iter__(self) -> Iterable[GraphBlock]:
        for batch in self.loader:
            num_nodes = batch.num_nodes 
            num_target_nodes = len(batch.input_id) 
            assert torch.all(batch.n_id[:num_target_nodes] == batch.input_id)
            target_node_mask = torch.zeros(num_nodes, dtype=torch.bool) 
            target_node_mask[:num_target_nodes] = True 
            
            yield GraphBlock(
                edge_index = batch.edge_index, 
                num_nodes = num_nodes, 
                num_target_nodes = num_target_nodes, 
                target_node_mask = target_node_mask,  
                target_node_idx = batch.input_id, 
                node_idx = batch.n_id, 
                edge_idx = batch.e_id, 
            )
