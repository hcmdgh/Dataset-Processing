import os 
import sys 


import pickle 

__all__ = [
    'convert_hg_to_triplets', 
]


def convert_hg_to_triplets():
    home_dir = os.getenv("HOME")
    dataset = DglNodePropPredDataset(
        name="ogbn-mag", root=os.path.join(home_dir, ".ogb", "dataset")
    )
    g, _ = dataset[0]

    node_type = g.ntypes
    node_offset = [0]
    for ntype in node_type:
        num_nodes = g.number_of_nodes(ntype)
        node_offset.append(num_nodes + node_offset[-1])

    node_offset = node_offset[:-1]

    with open(f"train_triplets_{args.dataset}", "w") as f:
        for etype in g.etypes:
            stype, _, dtype = g.to_canonical_etype(etype)
            src, dst = g.all_edges(etype=etype)
            src = src.numpy() + node_offset[node_type.index(stype)]
            dst = dst.numpy() + node_offset[node_type.index(dtype)]

            for u, v in zip(src, dst):
                f.write("{}\t{}\t{}\n".format(u, etype, v))
