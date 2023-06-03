from ..imports import * 
from ..metric import compute_acc 
from ..util import log_debug, log_info, log_warning, get_device
from ..dataloader import IndexLoader 

__all__ = [
    'MLP', 
]


class MLP(nn.Module):
    def __init__(self,
                 in_dim: int, 
                 out_dim: int, 
                 num_layers: int,
                 activation: nn.Module = nn.ReLU()):
        super().__init__() 
        
        self.num_layers = num_layers 
        
        hidden_dim_list = np.linspace(in_dim, out_dim, num=num_layers+1, dtype=np.int64) 

        self.fc_list = nn.ModuleList([
            nn.Linear(hidden_dim_list[l], hidden_dim_list[l+1])  
            for l in range(num_layers) 
        ])
        
        self.activation = activation 
        
    def forward(self,
                x: FloatTensor) -> FloatTensor: 
        h = x 
                
        for l in range(self.num_layers): 
            h = self.fc_list[l](h) 
            
            if l < self.num_layers - 1: 
                h = self.activation(h) 
        
        return h 
                     
    @staticmethod
    def classify(
        train_feat: FloatTensor,
        train_label: IntTensor,
        val_feat: FloatTensor,
        val_label: IntTensor,
        test_feat: FloatTensor,
        test_label: IntTensor,
        device: Any = None, 
        num_layers: int = 1,
        activation: nn.Module = nn.ReLU(),
        lr: float = 0.001,
        num_epochs: int = 500,
        batch_size: int = INF, 
        use_tqdm: bool = True,
        converge_warning: bool = True,  
    ) -> dict[str, Any]:
        if device is None:
            device = get_device() 
        else:
            device = torch.device(device) 
        
        train_feat = train_feat.detach().float().to(device) 
        train_label = train_label.detach().long().to(device) 
        val_feat = val_feat.detach().float().to(device) 
        val_label = val_label.detach().long().to(device) 
        test_feat = test_feat.detach().float().to(device) 
        test_label = test_label.detach().long().to(device) 
        
        feat_dim = train_feat.shape[-1]
        num_classes = int(torch.max(torch.cat([train_label, val_label, test_label]))) + 1 

        model = MLP(
            in_dim = feat_dim, 
            out_dim = num_classes, 
            num_layers = num_layers, 
            activation = activation, 
        )
        model = model.to(device)
        
        train_loader = IndexLoader(
            len(train_feat), 
            batch_size = batch_size, 
            shuffle = True, 
        )
        val_loader = IndexLoader(
            len(val_feat), 
            batch_size = batch_size, 
            shuffle = False, 
        )
        test_loader = IndexLoader(
            len(test_feat), 
            batch_size = batch_size, 
            shuffle = False, 
        )

        real_lr = lr / len(train_loader)
        optimizer = optim.Adam(model.parameters(), lr=real_lr) 
        
        epoch_to_val_acc: dict[int, float] = dict() 
        epoch_to_test_acc: dict[int, float] = dict()

        for epoch in tqdm(range(1, num_epochs + 1), disable=not use_tqdm, desc='MLP Classify', unit='epoch'):
            model.train() 
            
            for batch_idx in train_loader:
                pred_batch = model(train_feat[batch_idx]) 
                label_batch = train_label[batch_idx] 
                
                loss = F.cross_entropy(input=pred_batch, target=label_batch)
                
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step() 
            
            model.eval() 
            
            pred_batch_list = []
            label_batch_list = []
            
            for batch_idx in val_loader:
                with torch.no_grad():
                    pred_batch = model(val_feat[batch_idx]).detach().cpu()  
                    
                label_batch = val_label[batch_idx].detach().cpu() 
                
                pred_batch_list.append(pred_batch) 
                label_batch_list.append(label_batch)

            val_acc = compute_acc(
                pred = torch.cat(pred_batch_list, dim=0), 
                target = torch.cat(label_batch_list, dim=0), 
            )
            
            pred_batch_list = []
            label_batch_list = []
            
            for batch_idx in test_loader:
                with torch.no_grad():
                    pred_batch = model(test_feat[batch_idx]).detach().cpu()  
                    
                label_batch = test_label[batch_idx].detach().cpu() 
                
                pred_batch_list.append(pred_batch) 
                label_batch_list.append(label_batch)

            test_acc = compute_acc(
                pred = torch.cat(pred_batch_list, dim=0), 
                target = torch.cat(label_batch_list, dim=0), 
            )
            
            epoch_to_val_acc[epoch] = val_acc 
            epoch_to_test_acc[epoch] = test_acc 

        best_val_acc_epoch, best_val_acc = max(epoch_to_val_acc.items(), key=lambda x: (x[1], -x[0]))
        best_test_acc_epoch, best_test_acc = max(epoch_to_test_acc.items(), key=lambda x: (x[1], -x[0]))

        if best_val_acc_epoch >= num_epochs - 10 \
                or best_test_acc_epoch >= num_epochs - 10:
            if converge_warning:
                log_warning("The MLP model has not fully converged.")

        return dict(
            best_val_acc_epoch = best_val_acc_epoch,
            best_val_acc = best_val_acc,
            best_test_acc_epoch = best_test_acc_epoch,
            best_test_acc = best_test_acc,
        )
