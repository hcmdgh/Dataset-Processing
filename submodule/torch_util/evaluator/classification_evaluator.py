import wandb 

from ..imports import * 
from ..metric import compute_acc 
from ..util import log_debug, log_info


__all__ = [
    'ClassificationEvaluator', 
]


class ClassificationEvaluator:
    def __init__(self,
                 use_wandb: bool = False,
                 use_tensorboard: bool = False):
        self.use_wandb = use_wandb 
            
        if use_tensorboard:
            raise NotImplementedError
            self.summary_writer = SummaryWriter(log_dir='./runs')
        else:
            self.summary_writer = None 

        self.epoch_to_loss: dict[int, float] = dict()
        self.epoch_to_train_acc: dict[int, float] = dict() 
        self.epoch_to_val_acc: dict[int, float] = dict() 
        self.epoch_to_test_acc: dict[int, float] = dict()

    def compute_train_loss(self,
                           pred: FloatTensor,
                           target: IntTensor) -> FloatScalarTensor:
        loss = F.cross_entropy(input=pred, target=target)
        
        return loss 
    
    def record_train_loss(self,
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
    
    def compute_acc(self,
                    pred: FloatTensor,
                    target: IntTensor) -> float:
        acc = compute_acc(pred=pred, target=target)
        
        return acc
    
    def record_train_acc(self,
                         epoch: int, 
                         train_acc: float):
        assert epoch not in self.epoch_to_train_acc
        self.epoch_to_train_acc[epoch] = train_acc
        
        best_train_acc_epoch, best_train_acc = max(self.epoch_to_train_acc.items(), key=lambda x: (x[1], -x[0]))

        log_debug(f"[Train] Epoch: {epoch}, Train Acc: {train_acc:.4f} (Max: {best_train_acc:.4f} in Epoch {best_train_acc_epoch}).")

        if self.use_wandb:
            wandb.log(
                { 
                    'Train Acc': train_acc, 
                }, 
                step = epoch, 
            )
            
        if self.summary_writer is not None:
            self.summary_writer.add_scalar('Train Acc', train_acc, epoch) 

    def compute_and_record_train_acc(self,
                                     epoch: int,
                                     pred: FloatTensor,
                                     target: IntTensor) -> float:
        train_acc = self.compute_acc(pred=pred, target=target)

        self.record_train_acc(epoch=epoch, train_acc=train_acc) 
        
        return train_acc 
            
    def record_val_acc(self,
                       epoch: int, 
                       val_acc: float):
        assert epoch not in self.epoch_to_val_acc
        self.epoch_to_val_acc[epoch] = val_acc
        
        best_val_acc_epoch, best_val_acc = max(self.epoch_to_val_acc.items(), key=lambda x: (x[1], -x[0]))

        log_debug(f"[Val] Epoch: {epoch}, Val Acc: {val_acc:.4f} (Max: {best_val_acc:.4f} in Epoch {best_val_acc_epoch}).")

        if self.use_wandb:
            wandb.log(
                { 
                    'Val Acc': val_acc, 
                }, 
                step = epoch, 
            )
            
        if self.summary_writer is not None:
            self.summary_writer.add_scalar('Val Acc', val_acc, epoch) 

    def compute_and_record_val_acc(self,
                                   epoch: int,
                                   pred: FloatTensor,
                                   target: IntTensor) -> float:
        val_acc = self.compute_acc(pred=pred, target=target)

        self.record_val_acc(epoch=epoch, val_acc=val_acc) 
        
        return val_acc 
    
    def record_test_acc(self,
                        epoch: int, 
                        test_acc: float):
        assert epoch not in self.epoch_to_test_acc
        self.epoch_to_test_acc[epoch] = test_acc
        
        best_test_acc_epoch, best_test_acc = max(self.epoch_to_test_acc.items(), key=lambda x: (x[1], -x[0]))

        log_debug(f"[Test] Epoch: {epoch}, Test Acc: {test_acc:.4f} (Max: {best_test_acc:.4f} in Epoch {best_test_acc_epoch}).")

        if self.use_wandb:
            wandb.log(
                { 
                    'Test Acc': test_acc, 
                }, 
                step = epoch, 
            )
            
        if self.summary_writer is not None:
            self.summary_writer.add_scalar('Test Acc', test_acc, epoch) 
            
    def compute_and_record_test_acc(self,
                                    epoch: int,
                                    pred: FloatTensor,
                                    target: IntTensor) -> float:
        test_acc = self.compute_acc(pred=pred, target=target)

        self.record_test_acc(epoch=epoch, test_acc=test_acc) 
        
        return test_acc 
            
    def check_early_stopping(self,
                             num_tolerant_epochs: int = 50) -> bool:
        current_epoch = max(self.epoch_to_test_acc.keys()) 
        previous_epoch = current_epoch - num_tolerant_epochs 
        
        if previous_epoch <= 0:
            return False 
        
        assert previous_epoch in self.epoch_to_test_acc 
        
        previous_metric = self.epoch_to_test_acc[previous_epoch]
        history_metrics = [v for k, v in self.epoch_to_test_acc.items() if k >= previous_epoch]
        max_history_metric = max(history_metrics)
        
        if max_history_metric <= previous_metric:
            log_info(f"[Early Stopping] Epoch: {current_epoch}")
             
            return True 
        else:
            return False 

    def summary(self) -> dict[str, Any]:
        min_loss_epoch, min_loss = min(self.epoch_to_loss.items(), key=lambda x: (x[1], x[0]))
        max_train_acc_epoch, max_train_acc = max(self.epoch_to_train_acc.items(), key=lambda x: (x[1], -x[0]))
        max_val_acc_epoch, max_val_acc = max(self.epoch_to_val_acc.items(), key=lambda x: (x[1], -x[0]))
        max_test_acc_epoch, max_test_acc = max(self.epoch_to_test_acc.items(), key=lambda x: (x[1], -x[0]))

        log_info(
            "[Summary]\n"
            f"    Min Loss: {min_loss:.5f} (in Epoch {min_loss_epoch})\n"
            f"    Max Train Acc: {max_train_acc:.4f} (in Epoch {max_train_acc_epoch})\n"
            f"    Max Val Acc: {max_val_acc:.4f} (in Epoch {max_val_acc_epoch})\n"
            f"    Max Test Acc: {max_test_acc:.4f} (in Epoch {max_test_acc_epoch})\n"
        )

        if self.use_wandb:
            wandb.summary['Min Loss'] = min_loss 
            wandb.summary['Min Loss Epoch'] = min_loss_epoch  
            wandb.summary['Max Train Acc'] = max_train_acc  
            wandb.summary['Max Train Acc Epoch'] = max_train_acc_epoch  
            wandb.summary['Max Val Acc'] = max_val_acc  
            wandb.summary['Max Val Acc Epoch'] = max_val_acc_epoch  
            wandb.summary['Max Test Acc'] = max_test_acc  
            wandb.summary['Max Test Acc Epoch'] = max_test_acc_epoch  
            
            wandb.finish() 
            
        if self.summary_writer is not None: 
            self.summary_writer.close()

        return dict(
            min_loss = min_loss,
            min_loss_epoch = min_loss_epoch,
            max_train_acc = max_train_acc,
            max_train_acc_epoch = max_train_acc_epoch,
            max_val_acc = max_val_acc,
            max_val_acc_epoch = max_val_acc_epoch,
            max_test_acc = max_test_acc,
            max_test_acc_epoch = max_test_acc_epoch,
        )
