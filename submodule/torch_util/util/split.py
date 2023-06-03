from ..imports import * 

__all__ = [
    'split_train_val_test_set', 
    'split_train_val_set', 
]


def split_train_val_test_set(total_cnt: int,
                             train_ratio: float,
                             val_ratio: float) -> tuple[BoolArray, BoolArray, BoolArray]:
    test_ratio = 1. - train_ratio - val_ratio 
    assert test_ratio > 0. 
    
    train_cnt = int(total_cnt * train_ratio)
    val_cnt = int(total_cnt * val_ratio)
    train_mask = np.zeros(total_cnt, dtype=bool)
    val_mask = np.zeros(total_cnt, dtype=bool)
    test_mask = np.zeros(total_cnt, dtype=bool)
    perm = np.random.permutation(total_cnt)
    train_mask[perm[:train_cnt]] = True 
    val_mask[perm[train_cnt: train_cnt + val_cnt]] = True 
    test_mask[perm[train_cnt + val_cnt:]] = True 
    assert train_mask.sum() + val_mask.sum() + test_mask.sum() == total_cnt 
    
    return train_mask, val_mask, test_mask 


def split_train_val_set(full_set: Any,
                        train_ratio: float) -> tuple[BoolArray, BoolArray]: 
    if isinstance(full_set, int):
        full_mask = np.ones(full_set, dtype=bool) 
    elif isinstance(full_set, np.ndarray) and full_set.dtype == bool: 
        full_mask = full_set 
    elif isinstance(full_set, Tensor) and full_set.dtype == torch.bool: 
        full_mask = full_set.cpu().numpy() 
    else:
        raise TypeError 
    
    N, = full_mask.shape 
    train_mask = np.zeros(N, dtype=bool)
    val_mask = np.zeros(N, dtype=bool) 

    full_idx = full_mask.nonzero()[0] 
    M, = full_idx.shape 
    train_cnt = int(M * train_ratio) 
    
    perm = np.random.permutation(M) 
    train_mask[full_idx[perm[:train_cnt]]] = True 
    val_mask[full_idx[perm[train_cnt:]]] = True 
    
    assert train_mask.sum() + val_mask.sum() == M 
    assert np.all(~(train_mask & val_mask)) 
    assert np.all(train_mask | val_mask == full_mask)

    return train_mask, val_mask 
