import argparse
import networkx
import torch
import torch.nn as nn
import os
import torch.nn.functional as F

from tqdm import tqdm
from collections import defaultdict

from deep_event_mine import configdem
from deep_event_mine.nets import deepEM
from deep_event_mine.utils import utils
from deep_event_mine.eval.evalEV import write_events
from deep_event_mine.event_to_graph.standoff2graphs import get_graphs
from deep_event_mine.loader.prepData import prepdata


class DeepGraphMine(nn.Module):
    def __init__(self, config_file):
        super().__init__()
        self.params['device'] = "cuda:0"
        self.pred_params, self.params = configdem.config(config_file)
        self.device = self.params['device']
        self.deepee_model = deepEM.DeepEM(self.params)

        # load pretrained weights
        utils.handle_checkpoints(model=self.deepee_model,
                                 checkpoint_dir=self.params['model_path'],
                                 params={
                                     'device': self.device
                                 },
                                 resume=True)

        result_dir = self.pred_params['result_dir']
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        self.a2_files_path = self.params['result_dir'] + 'ev-last/ev-tok-ann/'
        self.mapping_id_tag = self.params['mappings']['nn_mapping']['id_tag_mapping']
        self.node_dim = self.params['bert_dim'] * 3 + self.params['etype_dim']

    def pad_nodes_adjs(self, nodes_input, adjs_input, max_len):

        nodes_output = []
        adjs_output = []
        for i, (node_input, adj_input) in enumerate(zip(nodes_input, adjs_input)):
            nodes_output.append(F.pad(node_input, pad=(0, 0, 0, max_len - node_input.shape[0])))
            adjs_output.append(F.pad(adj_input, pad=(0, max_len - node_input.shape[0], 0, max_len - node_input.shape[0])))

        return nodes_output, adjs_output

    def embeddings_for_entities(self, all_ner_terms, all_ent_embs, fidss, feid_mapping):

        # article ids of the documents in the batch
        articles_ids = set([item for sublist in fidss for item in sublist])

        # create a dictionary for each document in the batch
        # where we store the node embedding for each entity
        entities_dict = defaultdict(dict)
        for bb, ner_terms in enumerate(all_ner_terms):

                for sent, ner_term in enumerate(ner_terms):

                    if fidss[bb][sent] in feid_mapping:

                        for index, i_d in ner_term[0].items():
                            if i_d in feid_mapping[fidss[bb][sent]]:
                                node_emb = all_ent_embs[bb][sent][index]
                                correct_index = feid_mapping[fidss[bb][sent]][i_d]
                                entities_dict[fidss[bb][sent]][correct_index] = (sent, index, node_emb)

        # contains a n graphs for each document in the batch
        # constructed using, events triggers and entities
        all_graphs = get_graphs(self.a2_files_path, articles_ids)
        nodes_num = []
        for doc_id, graphs in all_graphs.items():

            # if the graph contains at least 1 event and it's not empty
            if len(graphs) > 0:
                full_graph = networkx.DiGraph(source_doc=doc_id)
                full_graph.add_node('master_node')

                for graph in graphs:
                    full_graph = networkx.compose(full_graph, graph)

                all_graphs[doc_id] = full_graph
                nodes_num.append(len(full_graph.nodes))

            else:
                full_graph = networkx.DiGraph(source_doc=doc_id)
                full_graph.add_node('master_node')
                nodes_num.append(0)
                all_graphs[doc_id] = full_graph

        init_nodes_vec = []
        init_adjs = []
        for doc_id, graph in all_graphs.items():
            nodes_emb = [torch.zeros(self.node_dim, device=self.device) if node_id == 'master_node' else entities_dict[doc_id][node_id][2] for node_id in graph.nodes]
            nodes_emb = torch.stack(nodes_emb)
            init_nodes_vec.append(nodes_emb)

            init_adjs.append(torch.from_numpy(networkx.adjacency_matrix(graph).todense()))

        init_nodes_vec, init_adjs = self.pad_nodes_adjs(init_nodes_vec, init_adjs, max(nodes_num))
        batch_nodes_vec = torch.stack(init_nodes_vec, 0)
        batch_adjs = torch.stack(init_adjs, 0)
        batch_adjs = batch_adjs.to(self.device, dtype=torch.float32)

        return batch_nodes_vec, batch_adjs, nodes_num

    def generate_events(self, nntrain_data, train_dataloader):

        # store predicted entities
        ent_preds = []

        # store predicted events
        ev_preds = []

        fidss, wordss, offsetss, sub_to_wordss, span_indicess = [], [], [], [], []
        all_ent_embs, all_ner_preds, all_ner_terms = [], [], []

        # entity and relation output
        ent_anns = []

        is_eval_ev = False
        for batch in tqdm(train_dataloader):
            eval_data_ids = batch
            tensors = utils.get_tensors(eval_data_ids, nntrain_data, self.params)

            nn_tokens, nn_ids, nn_token_mask, nn_attention_mask, nn_span_indices, nn_span_labels, nn_span_labels_match_rel, nn_entity_masks, nn_trigger_masks, _, \
            etypes, _ = tensors

            fids = [
                nntrain_data["fids"][data_id] for data_id in eval_data_ids[0].tolist()
            ]
            offsets = [
                nntrain_data["offsets"][data_id]
                for data_id in eval_data_ids[0].tolist()
            ]
            words = [
                nntrain_data["words"][data_id] for data_id in eval_data_ids[0].tolist()
            ]
            sub_to_words = [
                nntrain_data["sub_to_words"][data_id]
                for data_id in eval_data_ids[0].tolist()
            ]
            subwords = [
                nntrain_data["subwords"][data_id]
                for data_id in eval_data_ids[0].tolist()
            ]

            ner_out, rel_out, ev_out = self.deepee_model(tensors, self.params)

            ner_preds = ner_out['preds']

            ner_terms = ner_out['terms']

            all_ner_terms.append(ner_terms)

            for sentence_idx, ner_pred in enumerate(ner_preds):

                pred_entities = []
                for span_id, ner_pred_id in enumerate(ner_pred):
                    span_start, span_end = nn_span_indices[sentence_idx][span_id]
                    span_start, span_end = span_start.item(), span_end.item()
                    if (ner_pred_id > 0
                            and span_start in sub_to_words[sentence_idx]
                            and span_end in sub_to_words[sentence_idx]
                    ):
                        pred_entities.append(
                            (
                                sub_to_words[sentence_idx][span_start],
                                sub_to_words[sentence_idx][span_end],
                                self.mapping_id_tag[ner_pred_id],
                            )
                        )
                all_ner_preds.append(pred_entities)

            # entity prediction
            ent_ann = {'span_indices': nn_span_indices, 'ner_preds': ner_out['preds'], 'words': words,
                       'offsets': offsets, 'sub_to_words': sub_to_words, 'subwords': subwords,
                       'ner_terms': ner_terms}
            ent_anns.append(ent_ann)

            fidss.append(fids)

            wordss.append(words)
            offsetss.append(offsets)
            sub_to_wordss.append(sub_to_words)

            # relation prediction
            if rel_out['next']:
                all_ent_embs.append(rel_out['enttoks_type_embeds'])
            else:
                all_ent_embs.append([])

            # event prediction
            if ev_out is not None:
                # add predicted entity
                ent_preds.append(ner_out["nner_preds"])

                # add predicted events
                ev_preds.append(ev_out)

                span_indicess.append(
                    [
                        indice.detach().cpu().numpy()
                        for indice in ner_out["span_indices"]
                    ]
                )
                is_eval_ev = True
            else:
                ent_preds.append([])
                ev_preds.append([])

                span_indicess.append([])

        if is_eval_ev > 0:
            feid_mapping = write_events(fids=fidss,
                                        all_ent_preds=ent_preds,
                                        all_words=wordss,
                                        all_offsets=offsetss,
                                        all_span_terms=all_ner_terms,
                                        all_span_indices=span_indicess,
                                        all_sub_to_words=sub_to_wordss,
                                        all_ev_preds=ev_preds,
                                        g_entity_ids_=nntrain_data['g_entity_ids_'],
                                        params=self.params,
                                        result_dir=self.params['result_dir'])

            return all_ner_terms, all_ent_embs, fidss, feid_mapping
        else:
            return all_ner_terms, all_ent_embs, fidss, None

    def forward(self, sentences):

        train_data = prepdata.prep_input_data(None, self.params, sentences0=sentences)
        nntrain_data, train_dataloader = configdem.read_test_data(train_data, self.params)
        nntrain_data['g_entity_ids_'] = train_data['g_entity_ids_']

        all_ner_terms, all_ent_embs, fidss, feid_mapping = self.generate_events(nntrain_data, train_dataloader)

        if feid_mapping is not None:
            batch_nodes_vec, batch_adjs, nodes_num = self.embeddings_for_entities(all_ner_terms, all_ent_embs, fidss, feid_mapping)

            return batch_nodes_vec, batch_adjs, nodes_num

        else:
            return None, None, [0] * len(sentences)

