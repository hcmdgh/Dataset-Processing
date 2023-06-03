import wandb 

from ..imports import * 
from ..metric import compute_hits_at_k
from ..util import log_debug, log_info 


__all__ = [
    'HitsEvaluator', 
]


class HitsEvaluator:
    def __init__(self,
                 k: int, 
                 use_wandb: bool = False,
                 use_tensorboard: bool = False):
        self.use_wandb = use_wandb 
        self.k = k  
            
        if use_tensorboard:
            raise NotImplementedError
            self.summary_writer = SummaryWriter(log_dir='./runs')
        else:
            self.summary_writer = None 

        self.epoch_to_loss: dict[int, float] = dict()
        self.epoch_to_val_hits: dict[int, float] = dict() 
        self.epoch_to_test_hits: dict[int, float] = dict()
        
    def compute_train_loss(self,
                           pos_pred: FloatTensor,
                           neg_pred: FloatTensor,
                           logits: bool = False) -> FloatScalarTensor:
        num_samples = len(pos_pred) 
        assert pos_pred.shape == neg_pred.shape == (num_samples,) 
                         
        if not logits:
            loss = F.binary_cross_entropy(
                input = torch.cat([pos_pred, neg_pred]), 
                target = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)]),
            ) 
        else:
            loss = F.binary_cross_entropy_with_logits(
                input = torch.cat([pos_pred, neg_pred]), 
                target = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)]),
            ) 
            
        return loss 
    
    def record_train_epoch(self,
                           epoch: int, 
                           loss: Any):
        loss = float(loss)
        
        assert epoch not in self.epoch_to_loss
        self.epoch_to_loss[epoch] = loss
        min_loss_epoch, min_loss = min(self.epoch_to_loss.items(), key=lambda x: (x[1], x[0]))
        
        log_debug(f"[Train] Epoch: {epoch}, Loss: {loss:.5f} (Min: {min_loss:.5f} in Epoch {min_loss_epoch}).")
        
        if self.use_wandb:
            wandb.log(
                {
                    'Loss': loss, 
                }, 
                step = epoch, 
            )
            
        if self.summary_writer is not None:
            self.summary_writer.add_scalar('Loss', loss, epoch) 
    
    def compute_hits(self,
                     pos_pred: FloatTensor, 
                     neg_pred: FloatTensor,
                     logits: bool = False) -> float:
        pos_cnt, = pos_pred.shape 
        neg_cnt, = neg_pred.shape 

        if logits: 
            pos_pred = torch.sigmoid(pos_pred) 
            neg_pred = torch.sigmoid(neg_pred) 
            
        result = compute_hits_at_k(
            pos_score = pos_pred, 
            neg_score = neg_pred, 
            k = self.k,  
        )
        
        return result  
            
    def record_val_epoch(self,
                         epoch: int, 
                         val_hits: float):  
        assert epoch not in self.epoch_to_val_hits 
        self.epoch_to_val_hits[epoch] = val_hits 
        
        best_val_hits_epoch, best_val_hits = max(self.epoch_to_val_hits.items(), key=lambda x: (x[1], -x[0]))

        log_debug(f"[Val] Epoch: {epoch}, Val Hits@{self.k}: {val_hits:.4f} (Max: {best_val_hits:.4f} in Epoch {best_val_hits_epoch}).")

        if self.use_wandb:
            wandb.log(
                { 
                    f'Val Hits@{self.k}': val_hits, 
                }, 
                step = epoch, 
            )
            
        if self.summary_writer is not None:
            self.summary_writer.add_scalar(f'Val Hits@{self.k}', val_hits, epoch) 
    
    def record_test_epoch(self,
                          epoch: int, 
                          test_hits: float):  
        assert epoch not in self.epoch_to_test_hits 
        self.epoch_to_test_hits[epoch] = test_hits 
        
        best_test_hits_epoch, best_test_hits = max(self.epoch_to_test_hits.items(), key=lambda x: (x[1], -x[0]))

        log_debug(f"[Test] Epoch: {epoch}, Test Hits@{self.k}: {test_hits:.4f} (Max: {best_test_hits:.4f} in Epoch {best_test_hits_epoch}).")

        if self.use_wandb:
            wandb.log(
                { 
                    f'Test Hits@{self.k}': test_hits, 
                }, 
                step = epoch, 
            )
            
        if self.summary_writer is not None:
            self.summary_writer.add_scalar(f'Test Hits@{self.k}', test_hits, epoch) 
            
    def check_early_stopping(self,
                             num_tolerant_epochs: int = 50) -> bool:
        current_epoch = max(self.epoch_to_test_hits.keys()) 
        previous_epoch = current_epoch - num_tolerant_epochs 
        
        if previous_epoch <= 0:
            return False 
        
        assert previous_epoch in self.epoch_to_test_hits 
        
        previous_metric = self.epoch_to_test_hits[previous_epoch]
        history_metrics = [v for k, v in self.epoch_to_test_hits.items() if k >= previous_epoch]
        max_history_metric = max(history_metrics)
        
        if max_history_metric <= previous_metric:
            log_info(f"[Early Stopping] Epoch: {current_epoch}")
             
            return True 
        else:
            return False 

    def summary(self) -> dict[str, Any]:
        min_loss_epoch, min_loss = min(self.epoch_to_loss.items(), key=lambda x: (x[1], x[0]))
        max_val_hits_epoch, max_val_hits = max(self.epoch_to_val_hits.items(), key=lambda x: (x[1], -x[0]))
        max_test_hits_epoch, max_test_hits = max(self.epoch_to_test_hits.items(), key=lambda x: (x[1], -x[0]))

        log_info(
            "[Summary]\n"
            f"    Min Loss: {min_loss:.5f} (in Epoch {min_loss_epoch})\n"
            f"    Max Val Hits@{self.k}: {max_val_hits:.4f} (in Epoch {max_val_hits_epoch})\n"
            f"    Max Test Hits@{self.k}: {max_test_hits:.4f} (in Epoch {max_test_hits_epoch})\n"
        )

        if self.use_wandb:
            wandb.summary['Min Loss'] = min_loss 
            wandb.summary['Min Loss Epoch'] = min_loss_epoch  
            wandb.summary['Max Val Acc'] = max_val_hits  
            wandb.summary['Max Val Acc Epoch'] = max_val_hits_epoch  
            wandb.summary['Max Test Acc'] = max_test_hits  
            wandb.summary['Max Test Acc Epoch'] = max_test_hits_epoch  
            
            wandb.finish() 
            
        if self.summary_writer is not None: 
            self.summary_writer.close()

        return dict(
            min_loss = min_loss,
            min_loss_epoch = min_loss_epoch,
            max_val_hits = max_val_hits,
            max_val_hits_epoch = max_val_hits_epoch,
            max_test_hits = max_test_hits,
            max_test_hits_epoch = max_test_hits_epoch,
        )
