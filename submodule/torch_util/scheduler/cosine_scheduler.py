from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer 

from ..imports import * 

__all__ = [
    'CosineScheduler', 
]


class CosineScheduler(LambdaLR): 
    def __init__(self,
                 optimizer: Optimizer, 
                 num_epochs: int): 
        lr_fn = lambda epoch: ( 1 + np.cos( epoch * np.pi / num_epochs) ) * 0.5

        super().__init__(
            optimizer = optimizer, 
            lr_lambda = lr_fn, 
        )
