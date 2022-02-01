import argparse

import torch
import pandas as pd
from tqdm import tqdm

from deep_event_mine.loader.prepData import prepdata
from deep_event_mine import configdem
from deep_event_mine.eval.evalEV import write_events
from deep_event_mine.utils import utils
from deep_event_mine.nets import deepEM


def main(dataset):

    data = pd.read_csv(f'CDSR_data/{dataset}.csv')
    pred_params, params = configdem.config(f'{dataset}.yaml')

    all_sentences = {}

    i = 0
    for article, article_id in zip(data['source'], data['article_id']):

        if i == 10:
            break

        article = article.replace('e.g.', '').replace('i.e.', '')
        sentences = article.split('. ')
        all_sentences[article_id] = sentences

        i += 1

    _data = prepdata.prep_input_data(None, params, sentences0=all_sentences)
    nn_data, _dataloader = configdem.read_test_data(_data, params)
    nn_data['g_entity_ids_'] = _data['g_entity_ids_']

    generate_events(nn_data, _dataloader, params)


def generate_events(nn_data, _dataloader, params):

    deepee_model = deepEM.DeepEM(params)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    utils.handle_checkpoints(model=deepee_model,
                             checkpoint_dir=params['model_path'],
                             params={
                                 'device': device
                             },
                             resume=True)

    # store predicted entities
    ent_preds = []

    # store predicted events
    ev_preds = []

    fidss, wordss, offsetss, sub_to_wordss, span_indicess = [], [], [], [], []
    all_ent_embs, all_ner_preds, all_ner_terms = [], [], []

    # entity and relation output
    ent_anns = []

    mapping_id_tag = params['mappings']['nn_mapping']['id_tag_mapping']

    is_eval_ev = False
    for batch in tqdm(_dataloader):
        eval_data_ids = batch
        tensors = utils.get_tensors(eval_data_ids, nn_data, params)

        nn_tokens, nn_ids, nn_token_mask, nn_attention_mask, nn_span_indices, nn_span_labels, nn_span_labels_match_rel, nn_entity_masks, nn_trigger_masks, _, \
        etypes, _ = tensors

        fids = [
            nn_data["fids"][data_id] for data_id in eval_data_ids[0].tolist()
        ]
        offsets = [
            nn_data["offsets"][data_id]
            for data_id in eval_data_ids[0].tolist()
        ]
        words = [
            nn_data["words"][data_id] for data_id in eval_data_ids[0].tolist()
        ]
        sub_to_words = [
            nn_data["sub_to_words"][data_id]
            for data_id in eval_data_ids[0].tolist()
        ]
        subwords = [
            nn_data["subwords"][data_id]
            for data_id in eval_data_ids[0].tolist()
        ]

        ner_out, rel_out, ev_out = deepee_model(tensors, params)

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

    if is_eval_ev > 0:
        feid_mapping = write_events(fids=fidss,
                                    all_ent_preds=ent_preds,
                                    all_words=wordss,
                                    all_offsets=offsetss,
                                    all_span_terms=all_ner_terms,
                                    all_span_indices=span_indicess,
                                    all_sub_to_words=sub_to_wordss,
                                    all_ev_preds=ev_preds,
                                    g_entity_ids_=nn_data['g_entity_ids_'],
                                    params=params,
                                    result_dir=params['result_dir'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='training of the abstractor (ML)'
    )
    parser.add_argument('--dataset', required=True, help='target dataset for generating events')
    args = parser.parse_args()
    main(args.dataset)
