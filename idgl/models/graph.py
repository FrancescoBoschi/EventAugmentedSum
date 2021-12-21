import torch
import torch.nn as nn
import torch.nn.functional as F

from idgl.layers.graphlearn import GraphLearner
from idgl.layers.scalable_graphlearn import AnchorGraphLearner
from idgl.layers.anchor import AnchorGCN
from idgl.layers.common import dropout
from idgl.layers.gnn import GCN, GAT, GraphSAGE
from idgl.utils.generic_utils import to_cuda, create_mask, batch_normalize_adj
from idgl.utils.constants import VERY_SMALL_NUMBER


class Graph(nn.Module):
    def __init__(self, config, num_feat):
        super(Graph, self).__init__()
        self.config = config
        self.graph_learn = config['graph_learn']
        self.graph_metric_type = config['graph_metric_type']
        self.graph_module = config['graph_module']
        self.device = config['device']
        hidden_size = config['hidden_size']
        self.dropout = config['dropout']
        self.graph_skip_conn = config['graph_skip_conn']
        self.graph_include_self = config.get('graph_include_self', True)
        self.scalable_run = config.get('scalable_run', False)

        self.encoder = GAT(nfeat=num_feat,
                           nhid=hidden_size,
                           nclass=hidden_size,
                           dropout=self.dropout,
                           nheads=config.get('gat_nhead', 1),
                           alpha=config.get('gat_alpha', 0.2))

        if self.graph_learn:
            graph_learn_fun = AnchorGraphLearner if self.scalable_run else GraphLearner
            self.graph_learner = graph_learn_fun(num_feat, config['graph_learn_hidden_size'],
                                                 topk=config['graph_learn_topk'],
                                                 epsilon=config['graph_learn_epsilon'],
                                                 num_pers=config['graph_learn_num_pers'],
                                                 metric_type=config['graph_metric_type'],
                                                 device=self.device)

            self.graph_learner2 = graph_learn_fun(hidden_size,
                                                  config.get('graph_learn_hidden_size2',
                                                             config['graph_learn_hidden_size']),
                                                  topk=config.get('graph_learn_topk2', config['graph_learn_topk']),
                                                  epsilon=config.get('graph_learn_epsilon2',
                                                                     config['graph_learn_epsilon']),
                                                  num_pers=config['graph_learn_num_pers'],
                                                  metric_type=config['graph_metric_type'],
                                                  device=self.device)

            print('[ Graph Learner ]')
            if config['graph_learn_regularization']:
                print('[ Graph Regularization]')
        else:
            self.graph_learner = None
            self.graph_learner2 = None

    def learn_graph(self, graph_learner, node_features, graph_skip_conn=None, graph_include_self=False, init_adj=None,
                    anchor_features=None):
        if self.graph_learn:
            if self.scalable_run:
                node_anchor_adj = graph_learner(node_features, anchor_features)
                return node_anchor_adj

            else:
                # raw_adj: corresponds to A^(t)
                raw_adj = graph_learner(node_features)

                # f(A^(t))
                adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)

                if graph_skip_conn in (0, None):
                    if graph_include_self:
                        adj = adj + to_cuda(torch.eye(adj.size(0)), self.device)
                else:
                    # lamda*L^(0) + (1-lamda) * f(A^(t))
                    adj = graph_skip_conn * init_adj + (1 - graph_skip_conn) * adj

                return raw_adj, adj

        else:
            raw_adj = None
            adj = init_adj

            return raw_adj, adj

    def prepare_init_graph(self, adj, node_size, node_lens):

        # 1 if we don't exceed length 0 if we need to pad. size e.g. (16, 327)
        # e.g. context_mask[0] = [1., 1., ....., 0]
        nodes_mask = create_mask(node_lens, node_size, device=self.device)

        # L^(0)
        init_adj = self.compute_init_adj(adj, mask=nodes_mask)
        return nodes_mask, init_adj

    def compute_init_adj(self, adj, mask):
        adj_norm = batch_normalize_adj(adj, mask=mask)
        return adj_norm
