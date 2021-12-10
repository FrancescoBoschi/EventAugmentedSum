import argparse
from tqdm import tqdm
import itertools
from collections import defaultdict

from deep_event_mine import configdem
from deep_event_mine.nets import deepEM
from deep_event_mine.utils import utils
from deep_event_mine.eval.evalEV import write_events
from deep_event_mine.event_to_graph.standoff2graphs import get_graphs


def embeddings_for_entities(all_ner_terms, all_ent_embs, fidss, feid_mapping):
    entities_list = defaultdict(list)

    for bb, ner_terms in enumerate(all_ner_terms):

        for sent, ner_term in enumerate(ner_terms):

            for index, i_d in ner_term[0].items():
                if i_d in feid_mapping[fidss[bb][sent]]:
                    node_emb = all_ent_embs[bb][sent][index]
                    correct_index = feid_mapping[fidss[bb][sent]][i_d]
                    entities_list[fidss[bb][sent]].append((sent, index, correct_index, node_emb))

    return entities_list


def get_range(dictionary, begin, end):
    return dict(itertools.islice(dictionary.items(), begin, end))


def dem_forward():
    import pickle
    file_to_read = open("dumb_data/sssentences0.pickle", "rb")
    ssentence0 = pickle.load(file_to_read)
    file_to_read = open("dumb_data/parameters.pickle", "rb")
    parameters = pickle.load(file_to_read)

    ind_start = 0
    ind_end = 20

    deepee_model = deepEM.DeepEM(parameters)

    model_path = parameters['model_path']
    device = parameters['device']

    utils.handle_checkpoints(model=deepee_model,
                             checkpoint_dir=model_path,
                             params={
                                 'device': device
                             },
                             resume=True)

    deepee_model.to(device)

    while ind_end < len(ssentence0):
        print(f'ind_start: {ind_start} \t ind_end: {ind_end}')

        sentence0 = get_range(ssentence0, ind_start, ind_end)
        train_data, train_dataloader, parameters = configdem.config(args.configdem, sentence0)

        mapping_id_tag = parameters['mappings']['nn_mapping']['id_tag_mapping']

        # store predicted entities
        ent_preds = []

        # store predicted events
        ev_preds = []

        fidss, wordss, offsetss, sub_to_wordss, span_indicess = [], [], [], [], []
        all_ent_embs, all_ner_preds, all_ner_terms = [], [], []

        # entity and relation output
        ent_anns = []

        is_eval_ev = False
        for batch in tqdm(train_dataloader, desc="Iteration", leave=False):
            eval_data_ids = batch
            tensors = utils.get_tensors(eval_data_ids, train_data, parameters)

            nn_tokens, nn_ids, nn_token_mask, nn_attention_mask, nn_span_indices, nn_span_labels, nn_span_labels_match_rel, nn_entity_masks, nn_trigger_masks, _, \
            etypes, _ = tensors

            fids = [
                train_data["fids"][data_id] for data_id in eval_data_ids[0].tolist()
            ]
            offsets = [
                train_data["offsets"][data_id]
                for data_id in eval_data_ids[0].tolist()
            ]
            words = [
                train_data["words"][data_id] for data_id in eval_data_ids[0].tolist()
            ]
            sub_to_words = [
                train_data["sub_to_words"][data_id]
                for data_id in eval_data_ids[0].tolist()
            ]
            subwords = [
                train_data["subwords"][data_id]
                for data_id in eval_data_ids[0].tolist()
            ]

            ner_out, rel_out, ev_out = deepee_model(tensors, parameters)

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
                                mapping_id_tag[ner_pred_id],
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

        # fidss: each list in the fidss list is of length batch_size e.g 16 containing all document names corresponding to each sentence in the batch [PMID-10473104, PMID-10473104, ......]
        # wordss: each list in the wordss list is of length batch_size containing a lists of all the words in each sentence
        # offsetss: each list in the offsetss list is of length batch_size containing a list of the offsets corresponding to each word in the sentence [[0, 3], [4, 7], .....]
        # all_ner_terms: each list in the offsetss list is of length batch_size containing id2label, id2term, term2id info
        if is_eval_ev > 0:
            feid_mapping = write_events(fids=fidss,
                                        all_ent_preds=ent_preds,
                                        all_words=wordss,
                                        all_offsets=offsetss,
                                        all_span_terms=all_ner_terms,
                                        all_span_indices=span_indicess,
                                        all_sub_to_words=sub_to_wordss,
                                        all_ev_preds=ev_preds,
                                        g_entity_ids_=train_data['g_entity_ids_'],
                                        params=parameters,
                                        result_dir=parameters['result_dir'])

        ind_start += 20
        ind_end += 20

    entities_list = embeddings_for_entities(all_ner_terms, all_ent_embs, fidss, feid_mapping)
    all_graphs = get_graphs(parameters['result_dir'] + 'ev-last/ev-tok-ann/')


def main():
    dem_forward()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training of the abstractor (ML)'
    )

    parser.add_argument('--configdem', required=True, help='DEM config file name')

    args = parser.parse_args()

    main()
