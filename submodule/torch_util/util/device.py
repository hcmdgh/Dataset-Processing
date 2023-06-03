from ..imports import * 
from .logging import log_info 

__all__ = [
    'auto_set_device', 
    'set_device', 
    'get_device', 
    'to_device', 
]

_device: torch.device = torch.device('cpu') 


def get_device() -> torch.device:
    return _device 
    
    
def set_device(device: Any) -> torch.device: 
    global _device 
    _device = torch.device(device)
    
    return _device 


def auto_set_device(use_gpu: bool = True) -> torch.device:
    if not use_gpu:
        return set_device('cpu')
    else:
        exe_res = os.popen('gpustat --json').read() 
        state_dict = json.loads(exe_res)
        gpu_infos = [] 
        
        for gpu_entry in state_dict['gpus']:
            gpu_id = int(gpu_entry['index'])
            used_mem = int(gpu_entry['memory.used'])
            gpu_infos.append((used_mem, gpu_id))
        
        gpu_infos.sort()
        device = f'cuda:{gpu_infos[0][1]}'
    
        log_info(f"The device is set as {device}.")
        
        return set_device(device)


def to_device(obj: Any,
              device: Any = None) -> Any:
    if device: 
        device = torch.device(device)
    else:
        device = get_device()
    
    if isinstance(obj, list):
        return [
            to_device(item)
            for item in obj 
        ]
    elif isinstance(obj, dict):
        return {
            key: to_device(value)
            for key, value in obj.items() 
        } 
    elif isinstance(obj, Tensor): 
        return obj.to(device) 
    elif isinstance(obj, ndarray):
        return torch.from_numpy(obj).to(device)
    elif isinstance(obj, dglsp.SparseMatrix): 
        return obj.to(device)
    else:
        raise TypeError  
