import os 
import sys 
os.chdir(os.path.dirname(__file__))

from dgl_wrapper import * 


def main():
    dataset = homo_graph_dataset.OGB.OgbnArxiv(add_self_loop=False, to_undirected=False)
    
    g = dataset.g 
    
    print(g)
    
    
if __name__ == '__main__':
    main() 
