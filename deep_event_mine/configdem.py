import torch
import pickle
import os
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from deep_event_mine.utils import utils
from deep_event_mine.loader.prepData import prepdata
from deep_event_mine.loader.prepNN import prep4nn


def read_test_data(test_data, params):
    test = prep4nn.data2network(test_data, 'predict', params)

    if len(test) == 0:
        raise ValueError("Test set empty.")

    test_data = prep4nn.torch_data_2_network(cdata2network=test, params=params, do_get_nn_data=True)

    # number of sentences
    te_data_size = len(test_data['nn_data']['ids'])

    test_data_ids = TensorDataset(torch.arange(te_data_size))
    test_sampler = SequentialSampler(test_data_ids)
    test_dataloader = DataLoader(test_data_ids, sampler=test_sampler, batch_size=params['batchsize'])
    return test_data, test_dataloader


def config(config_file, sentences0):
    config_path = 'deep_event_mine/configs/{}'.format(config_file)

    with open(config_path, 'r') as stream:
        pred_params = utils._ordered_load(stream)

    # Load pre-trained parameters
    with open(pred_params['saved_params'], "rb") as f:
        parameters = pickle.load(f)

    # build l2r_pairs
    parameters['predict'] = True

    # Set predict settings value for params
    parameters['gpu'] = pred_params['gpu']
    parameters['batchsize'] = pred_params['batchsize']
    if parameters['gpu'] >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(parameters['gpu']) if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(parameters['gpu'])
    else:
        device = torch.device("cpu")

    parameters['device'] = device
    parameters['train_data'] = pred_params['train_data']
    parameters['freeze_bert'] = pred_params['freeze_bert']
    parameters['compute_metrics'] = pred_params['compute_metrics']
    parameters['bert_model'] = pred_params['bert_model']
    parameters['result_dir'] = pred_params['result_dir']
    parameters['model_path'] = pred_params['model_path']
    parameters['raw_text'] = pred_params['raw_text']
    parameters['ner_predict_all'] = pred_params['ner_predict_all']
    parameters['compute_dem_loss'] = pred_params['compute_dem_loss']
    parameters['a2_entities'] = pred_params['a2_entities']

    result_dir = pred_params['result_dir']
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # process train data
    train_data = prepdata.prep_input_data(pred_params['train_data'], parameters, sentences0=sentences0)
    nntrain_data, train_dataloader = read_test_data(train_data, parameters)
    nntrain_data['g_entity_ids_'] = train_data['g_entity_ids_']

    return nntrain_data, train_dataloader, parameters,

