import argparse 
from typing import Type, TypeVar, Optional, Any 

__all__ = [
    'parse_args', 
]

T = TypeVar('T')


def parse_args(cls: Type[T],
               description: Optional[str] = None) -> T: 
    parser = argparse.ArgumentParser(description=description) 
    
    bool_arg_name_set = set() 
    
    for arg_name, arg_type in cls.__annotations__.items(): 
        default = getattr(cls, arg_name, None) 
        
        if arg_type == str: 
            parser.add_argument(f"--{arg_name}", type=str, required=default is None, default=default)
        elif arg_type == int: 
            parser.add_argument(f"--{arg_name}", type=int, required=default is None, default=default)
        elif arg_type == float: 
            parser.add_argument(f"--{arg_name}", type=float, required=default is None, default=default)
        elif arg_type == bool:
            if default == True: 
                default = 'true' 
            elif default == False: 
                default = 'false' 
             
            parser.add_argument(f"--{arg_name}", type=str, required=default is None, default=default, choices=['true', 'false']) 

            bool_arg_name_set.add(arg_name) 
        else:
            raise TypeError 
        
    args = parser.parse_args() 
    
    for arg_name in bool_arg_name_set: 
        if getattr(args, arg_name) == 'true': 
            setattr(args, arg_name, True)
        elif getattr(args, arg_name) == 'false': 
            setattr(args, arg_name, False)
        else:
            raise AssertionError 
    
    param_dict = {
        arg_name: getattr(args, arg_name)
        for arg_name in cls.__annotations__
    } 
    
    return cls(**param_dict) 
