import torch
import torch.nn as nn
import torch.nn.functional as F

from models.deepGM import DeepGraphMine
from idgl.models.graph import Graph
from idgl.utils.generic_utils import to_cuda


class EventAugmentedSumm(nn.Module):
    def __init__(self, configdgm, configIDGL):
        super().__init__()

        self.deepGM = DeepGraphMine(configdgm)
        self.IDGLnetwork = Graph(configIDGL, self.deepGM.node_dim)

    def forward(self, article, art_lens, abstract, extend_art, extend_vsize, sentences, fids):

        batch_nodes_vec, batch_adjs, nodes_num = self.deepGM(sentences)

    def batch_IGL(self, init_adj, init_node_vec, nodes_num, training, out_predictions=False):
        """
        :param init_adj: is the adjancency matrix A^(0). size (batch_size, num_nodes, num_nodes) e.g. (5, 2708, 2708)
        :param init_node_vec: it's the matrix X that contains the feature vectors for each node. size (num_nodes, num_feat) e.g. (2708, 2604)
        :param nodes_num: contains the actual number of nodes in each document. size (batch_size,) e.g. (5,)
        :param training: we are in traing mode if True
        :param out_predictions: save the predictions if True
        """

        # we set the model to training mod if training=True
        mode = "train" if training else ("test" if self.is_test else "dev")
        network = self.IDGLmodel.network
        network.train(training)

        # norm_init_adj: is the normalized adjacency matrix L^0. size (batch_size, num_nodes, num_nodes) e.g. (5, 2708, 2708)
        norm_init_adj, node_mask = network.prepare_init_graph(init_adj, init_node_vec.size(-1), nodes_num)

        # curr_raw_adj: corresponds to A^(1) (num_nodes, num_nodes) e.g. (2708, 2708)
        # cur_adj: corresponds to A^~(1) (num_nodes, num_nodes) e.g. (2708, 2708)
        cur_raw_adj, cur_adj = network.learn_graph(network.graph_learner, init_node_vec, network.graph_skip_conn, node_mask=node_mask, graph_include_self=network.graph_include_self, init_adj=norm_init_adj)

        # applying GAT
        node_vec = network.encoder(init_node_vec, cur_adj)
        node_vec = F.dropout(node_vec, network.dropout, training=network.training)

        loss1 = self.model.criterion(output, targets)
        score = self.model.score_func(targets.cpu(), output.detach().cpu())

        if self.config['graph_learn'] and self.config['graph_learn_regularization']:
            loss1 += self.add_batch_graph_loss(cur_raw_adj, init_node_vec)

        # first_raw_adj: corresponds to A^(1) (num_nodes, num_nodes) e.g. (2708, 2708)
        # first_adj: corresponds to A^~(1) (num_nodes, num_nodes) e.g. (2708, 2708)
        first_raw_adj, first_adj = cur_raw_adj, cur_adj

        # selecting number of iterations of Algorithm 1 by default it's 10
        if not mode == 'test':
            if self._epoch > self.config.get('pretrain_epoch', 0):
                max_iter_ = self.config.get('max_iter', 10) # Fine-tuning
                if self._epoch == self.config.get('pretrain_epoch', 0) + 1:
                    for k in self._dev_metrics:
                        self._best_metrics[k] = -float('inf')

            else:
                max_iter_ = 0 # Pretraining
        else:
            max_iter_ = self.config.get('max_iter', 10)

        # eps_adj: delta in the paper
        eps_adj = float(self.config.get('eps_adj', 0)) if training else float(self.config.get('test_eps_adj', self.config.get('eps_adj', 0)))

        loss = 0
        iter_ = 0

        # Indicate the last iteration number for each example
        batch_last_iters = to_cuda(torch.zeros(x_batch['batch_size'], dtype=torch.uint8), self.device)
        # Indicate either an example is in onging state (i.e., 1) or stopping state (i.e., 0)
        batch_stop_indicators = to_cuda(torch.ones(x_batch['batch_size'], dtype=torch.uint8), self.device)
        batch_all_outputs = []
        while self.config['graph_learn'] and (iter_ == 0 or torch.sum(batch_stop_indicators).item() > 0) and iter_ < max_iter_:
            iter_ += 1
            batch_last_iters += batch_stop_indicators

            # A^(t-1) (num_nodes, num_nodes) e.g. (2708, 2708)
            pre_raw_adj = cur_raw_adj

            # cur_raw_adj: corresponds to A^(t) (num_nodes, num_nodes) e.g. (2708, 2708)
            # cur_adj: corresponds to A^~(t) (num_nodes, num_nodes) e.g. (2708, 2708)
            cur_raw_adj, cur_adj = network.learn_graph(network.graph_learner2, node_vec, network.graph_skip_conn, node_mask=node_mask, graph_include_self=network.graph_include_self, init_adj=norm_init_adj)

            update_adj_ratio = self.config.get('update_adj_ratio', None)
            if update_adj_ratio is not None:
                cur_adj = update_adj_ratio * cur_adj + (1 - update_adj_ratio) * first_adj

            # apply GAT
            node_vec = network.encoder(init_node_vec, cur_adj)
            node_vec = F.dropout(node_vec, self.config.get('gl_dropout', 0), training=network.training)

            batch_all_outputs.append(tmp_output.unsqueeze(1))

            tmp_loss = self.model.criterion(tmp_output, targets, reduction='none')
            if len(tmp_loss.shape) == 2:
                tmp_loss = torch.mean(tmp_loss, 1)

            loss += batch_stop_indicators.float() * tmp_loss

            if self.config['graph_learn'] and self.config['graph_learn_regularization']:
                # adding L_G^(t) obtaining L^(t)
                loss += batch_stop_indicators.float() * self.add_batch_graph_loss(cur_raw_adj, init_node_vec, keep_batch_dim=True)

            if self.config['graph_learn'] and not self.config.get('graph_learn_ratio', None) in (None, 0):
                loss += batch_stop_indicators.float() * batch_SquaredFrobeniusNorm(cur_raw_adj - pre_raw_adj) * self.config.get('graph_learn_ratio')

            tmp_stop_criteria = batch_diff(cur_raw_adj, pre_raw_adj, first_raw_adj) > eps_adj
            batch_stop_indicators = batch_stop_indicators * tmp_stop_criteria

        if iter_ > 0:
            loss = torch.mean(loss / batch_last_iters.float()) + loss1

            batch_all_outputs = torch.cat(batch_all_outputs, 1)
            selected_iter_index = batch_last_iters.long().unsqueeze(-1) - 1

            if len(batch_all_outputs.shape) == 3:
                selected_iter_index = selected_iter_index.unsqueeze(-1).expand(-1, -1, batch_all_outputs.size(-1))
                output = batch_all_outputs.gather(1, selected_iter_index).squeeze(1)
            else:
                output = batch_all_outputs.gather(1, selected_iter_index)

            score = self.model.score_func(targets.cpu(), output.detach().cpu())


        else:
            loss = loss1

        res = {'loss': loss.item(),
                'metrics': {'nloss': -loss.item(), self.model.metric_name: score},
        }
        if out_predictions:
            res['predictions'] = output.detach().cpu()

        if training:
            loss = loss / self.config['grad_accumulated_steps'] # Normalize our loss (if averaged)
            loss.backward()

            if (step + 1) % self.config['grad_accumulated_steps'] == 0: # Wait for several backward steps
                self.model.clip_grad()
                self.model.optimizer.step()
                self.model.optimizer.zero_grad()
        return res
