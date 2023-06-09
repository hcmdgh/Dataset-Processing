from collections import defaultdict 


class Graph:
    def __init__(self):
        super(Graph, self).__init__()
        """
            node_forward and bacward are only used when building the data.
            Afterwards will be transformed into node_feature by DataFrame

            node_forward: name -> node_id
            node_bacward: node_id -> feature_dict
            node_feature: a DataFrame containing all features
        """
        self.node_forward = defaultdict(lambda: {})
        self.node_bacward = defaultdict(lambda: [])
        self.node_feature = defaultdict(lambda: [])

        """
            edge_list: index the adjacancy matrix (time) by
            <target_type, source_type, relation_type, target_id, source_id>
        """
        self.edge_list = defaultdict(  # target_type
            lambda: defaultdict(  # source_type
                lambda: defaultdict(  # relation_type
                    lambda: defaultdict(  # target_id
                        lambda: defaultdict(lambda: int)  # source_id(  # time
                    )
                )
            )
        )
        self.times = {}
