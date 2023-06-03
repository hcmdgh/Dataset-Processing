import dill 
from collections import defaultdict
import csv 
from tqdm import tqdm 
import json 


class Graph:
    def __init__(self):
        '''
            node_forward and bacward are only used when building the data. 
            Afterwards will be transformed into node_feature by DataFrame
            
            node_forward: name -> node_id
            node_bacward: node_id -> feature_dict
            node_feature: a DataFrame containing all features
        '''
        self.node_forward = defaultdict(lambda: {})
        self.node_bacward = defaultdict(lambda: [])
        self.node_feature = defaultdict(lambda: [])

        '''
            edge_list: index the adjacancy matrix (time) by 
            <target_type, source_type, relation_type, target_id, source_id>
        '''
        self.edge_list = defaultdict( #target_type
                            lambda: defaultdict(  #source_type
                                lambda: defaultdict(  #relation_type
                                    lambda: defaultdict(  #target_id
                                        lambda: defaultdict( #source_id(
                                            lambda: int # time
                                        )))))
        self.times = {}
    def add_node(self, node):
        nfl = self.node_forward[node['type']]
        if node['id'] not in nfl:
            self.node_bacward[node['type']] += [node]
            ser = len(nfl)
            nfl[node['id']] = ser
            return ser
        return nfl[node['id']]
    def add_edge(self, source_node, target_node, time = None, relation_type = None, directed = True):
        edge = [self.add_node(source_node), self.add_node(target_node)]
        '''
            Add bi-directional edges with different relation type
        '''
        self.edge_list[target_node['type']][source_node['type']][relation_type][edge[1]][edge[0]] = time
        if directed:
            self.edge_list[source_node['type']][target_node['type']]['rev_' + relation_type][edge[0]][edge[1]] = time
        else:
            self.edge_list[source_node['type']][target_node['type']][relation_type][edge[0]][edge[1]] = time
        self.times[time] = True
        
    def update_node(self, node):
        nbl = self.node_bacward[node['type']]
        ser = self.add_node(node)
        for k in node:
            if k not in nbl[ser]:
                nbl[ser][k] = node[k]

    def get_meta_graph(self):
        types = self.get_types()
        metas = []
        for target_type in self.edge_list:
            for source_type in self.edge_list[target_type]:
                for r_type in self.edge_list[target_type][source_type]:
                    metas += [(target_type, source_type, r_type)]
        return metas
    
    def get_types(self):
        return list(self.node_feature.keys())


class RenameUnpickler(dill.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "pyHGT.data" or module == 'data':
            renamed_module = "preprocess"
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


if __name__ == '__main__':
    print("正在加载OAG数据集......")
    graph = renamed_load(open('/Dataset/OAG-from-PTHGNN/Raw/graph_CS_20190919.pk', 'rb'))
    print("加载完毕！")
    
    # 抽取边关系
    with open('/Dataset/OAG-from-PTHGNN/Processed/CS/edge.csv', 'w', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=['src_ntype', 'etype', 'dest_ntype', 'src_nid', 'dest_nid'])
        writer.writeheader()
        
        # paper - PP - paper
        for src_nid, dest_nids in tqdm(graph.edge_list['paper']['paper']['PP_cite'].items()):
            for dest_nid in dest_nids:
                src_nid, dest_nid = int(src_nid), int(dest_nid)
                writer.writerow(
                    dict(
                        src_ntype = 'paper',
                        etype = 'PP', 
                        dest_ntype = 'paper',
                        src_nid = src_nid,
                        dest_nid = dest_nid, 
                    )
                )
                
        # paper - PV_conference - venue
        for src_nid, dest_nids in tqdm(graph.edge_list['paper']['venue']['rev_PV_Conference'].items()):
            for dest_nid in dest_nids:
                src_nid, dest_nid = int(src_nid), int(dest_nid)
                writer.writerow(
                    dict(
                        src_ntype = 'paper',
                        etype = 'PV_conference', 
                        dest_ntype = 'venue',
                        src_nid = src_nid,
                        dest_nid = dest_nid, 
                    )
                )
                
        # paper - PV_journal - venue
        for src_nid, dest_nids in tqdm(graph.edge_list['paper']['venue']['rev_PV_Journal'].items()):
            for dest_nid in dest_nids:
                src_nid, dest_nid = int(src_nid), int(dest_nid)
                writer.writerow(
                    dict(
                        src_ntype = 'paper',
                        etype = 'PV_journal', 
                        dest_ntype = 'venue',
                        src_nid = src_nid,
                        dest_nid = dest_nid, 
                    )
                )
                
        # paper - PF_L0 - field
        for src_nid, dest_nids in tqdm(graph.edge_list['paper']['field']['rev_PF_in_L0'].items()):
            for dest_nid in dest_nids:
                src_nid, dest_nid = int(src_nid), int(dest_nid)
                writer.writerow(
                    dict(
                        src_ntype = 'paper',
                        etype = 'PF_L0', 
                        dest_ntype = 'field',
                        src_nid = src_nid,
                        dest_nid = dest_nid, 
                    )
                )
                
        # paper - PF_L1 - field
        for src_nid, dest_nids in tqdm(graph.edge_list['paper']['field']['rev_PF_in_L1'].items()):
            for dest_nid in dest_nids:
                src_nid, dest_nid = int(src_nid), int(dest_nid)
                writer.writerow(
                    dict(
                        src_ntype = 'paper',
                        etype = 'PF_L1', 
                        dest_ntype = 'field',
                        src_nid = src_nid,
                        dest_nid = dest_nid, 
                    )
                )
                
        # paper - PF_L2 - field
        for src_nid, dest_nids in tqdm(graph.edge_list['paper']['field']['rev_PF_in_L2'].items()):
            for dest_nid in dest_nids:
                src_nid, dest_nid = int(src_nid), int(dest_nid)
                writer.writerow(
                    dict(
                        src_ntype = 'paper',
                        etype = 'PF_L2', 
                        dest_ntype = 'field',
                        src_nid = src_nid,
                        dest_nid = dest_nid, 
                    )
                )
                
        # paper - PF_L3 - field
        for src_nid, dest_nids in tqdm(graph.edge_list['paper']['field']['rev_PF_in_L3'].items()):
            for dest_nid in dest_nids:
                src_nid, dest_nid = int(src_nid), int(dest_nid)
                writer.writerow(
                    dict(
                        src_ntype = 'paper',
                        etype = 'PF_L3', 
                        dest_ntype = 'field',
                        src_nid = src_nid,
                        dest_nid = dest_nid, 
                    )
                )
                
        # paper - PF_L4 - field
        for src_nid, dest_nids in tqdm(graph.edge_list['paper']['field']['rev_PF_in_L4'].items()):
            for dest_nid in dest_nids:
                src_nid, dest_nid = int(src_nid), int(dest_nid)
                writer.writerow(
                    dict(
                        src_ntype = 'paper',
                        etype = 'PF_L4', 
                        dest_ntype = 'field',
                        src_nid = src_nid,
                        dest_nid = dest_nid, 
                    )
                )
                
        # paper - PF_L5 - field
        for src_nid, dest_nids in tqdm(graph.edge_list['paper']['field']['rev_PF_in_L5'].items()):
            for dest_nid in dest_nids:
                src_nid, dest_nid = int(src_nid), int(dest_nid)
                writer.writerow(
                    dict(
                        src_ntype = 'paper',
                        etype = 'PF_L5', 
                        dest_ntype = 'field',
                        src_nid = src_nid,
                        dest_nid = dest_nid, 
                    )
                )
                
        # paper - PA_first - author
        for src_nid, dest_nids in tqdm(graph.edge_list['paper']['author']['AP_write_first'].items()):
            for dest_nid in dest_nids:
                src_nid, dest_nid = int(src_nid), int(dest_nid)
                writer.writerow(
                    dict(
                        src_ntype = 'paper',
                        etype = 'PA_first', 
                        dest_ntype = 'author',
                        src_nid = src_nid,
                        dest_nid = dest_nid, 
                    )
                )
                
        # paper - PA_other - author
        for src_nid, dest_nids in tqdm(graph.edge_list['paper']['author']['AP_write_other'].items()):
            for dest_nid in dest_nids:
                src_nid, dest_nid = int(src_nid), int(dest_nid)
                writer.writerow(
                    dict(
                        src_ntype = 'paper',
                        etype = 'PA_other', 
                        dest_ntype = 'author',
                        src_nid = src_nid,
                        dest_nid = dest_nid, 
                    )
                )
                
        # paper - PA_last - author
        for src_nid, dest_nids in tqdm(graph.edge_list['paper']['author']['AP_write_last'].items()):
            for dest_nid in dest_nids:
                src_nid, dest_nid = int(src_nid), int(dest_nid)
                writer.writerow(
                    dict(
                        src_ntype = 'paper',
                        etype = 'PA_last', 
                        dest_ntype = 'author',
                        src_nid = src_nid,
                        dest_nid = dest_nid, 
                    )
                )
                
        # author - AI - institution
        for src_nid, dest_nids in tqdm(graph.edge_list['author']['affiliation']['rev_in'].items()):
            for dest_nid in dest_nids:
                src_nid, dest_nid = int(src_nid), int(dest_nid)
                writer.writerow(
                    dict(
                        src_ntype = 'author',
                        etype = 'AI', 
                        dest_ntype = 'institution',
                        src_nid = src_nid,
                        dest_nid = dest_nid, 
                    )
                )
                
        # field - FF - field
        for src_nid, dest_nids in tqdm(graph.edge_list['field']['field']['FF_in'].items()):
            for dest_nid in dest_nids:
                src_nid, dest_nid = int(src_nid), int(dest_nid)
                writer.writerow(
                    dict(
                        src_ntype = 'field',
                        etype = 'FF', 
                        dest_ntype = 'field',
                        src_nid = src_nid,
                        dest_nid = dest_nid, 
                    )
                )

    # 抽取结点特征
    with open('/Dataset/OAG-from-PTHGNN/Processed/CS/node_attr_paper.tsv', 'w', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, delimiter='\t', fieldnames=['nid', 'raw_id', 'title', 'year', 'citation_count', 'embed'])
        writer.writeheader()
        
        for nid, row in enumerate(tqdm(graph.node_feature['paper'].itertuples())):
            raw_id = int(row.id)
            title = row.title 
            year = int(row.time)
            embed = row.emb 
            assert isinstance(embed, list) and isinstance(embed[0], float)
            embed_str = json.dumps(embed, ensure_ascii=False).strip() 
            citation_count = int(row.citation)
            assert row.type == 'paper'
            
            writer.writerow(
                dict(
                    nid = nid, 
                    raw_id = raw_id, 
                    title = title, 
                    year = year, 
                    citation_count = citation_count,
                    embed = embed_str,  
                )
            )

    with open('/Dataset/OAG-from-PTHGNN/Processed/CS/node_attr_venue.tsv', 'w', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, delimiter='\t', fieldnames=['nid', 'raw_id', 'type', 'name', 'citation_count', 'embed_1', 'embed_2'])
        writer.writeheader()
        
        for nid, row in enumerate(tqdm(graph.node_feature['paper'].itertuples())):
            raw_id = int(row.id)
            type_ = row.attr 
            name = row.name
            citation_count = int(row.citation)  
            embed_1 = [float(x) for x in row.emb.tolist()] 
            embed_2 = [float(x) for x in row.node_emb.tolist()] 
            embed_1_str = json.dumps(embed_1, ensure_ascii=False).strip()
            embed_2_str = json.dumps(embed_2, ensure_ascii=False).strip()
            assert row.type == 'venue'
            
            writer.writerow(
                dict(
                    nid = nid, 
                    raw_id = raw_id, 
                    type = type_, 
                    name = name, 
                    citation_count = citation_count, 
                    embed_1 = embed_1_str,
                    embed_2 = embed_2_str,
                )
            )
