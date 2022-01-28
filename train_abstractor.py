""" train the abstractor"""
import pandas as pd

from graph_augmented_sum.training import get_basic_grad_fn, basic_validate
import argparse
import json
import os, re
from os.path import join, exists
import pickle as pkl

from cytoolz import compose, concat
from transformers import RobertaTokenizer
from tqdm import tqdm

import torch
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from models.eventAS import EventAugmentedSumm
from graph_augmented_sum.model.util import sequence_loss

from graph_augmented_sum.data.batcher import coll_fn, prepro_fn
from graph_augmented_sum.data.batcher import prepro_fn_copy_bert, convert_batch_copy_bert, batchify_fn_copy_bert
from graph_augmented_sum.data.batcher import BucketedGenerater

import pickle


class CDSRDataset(Dataset):
    """ single article sentence -> single abstract sentence
    (dataset created by greedily matching ROUGE)
    """

    def __init__(self, split: str, path: str):
        self._data_path = join(path, '{}.csv'.format(split))

        self._data_df = pd.read_csv(self._data_path)

    def __len__(self):
        return len(self._data_df)

    def __getitem__(self, i):
        target = self._data_df.loc[i, 'target']
        source = self._data_df.loc[i, 'source']
        article_id = self._data_df.loc[i, 'article_id']

        return source, target, article_id


def configure_net(configdgm, configIDGL, vocab_size, emb_dim,
                  n_hidden, bidirectional, n_layer, batch_size, bert_length):
    csg_net_args = {}
    csg_net_args['vocab_size'] = vocab_size
    csg_net_args['emb_dim'] = emb_dim
    csg_net_args['side_dim'] = n_hidden
    csg_net_args['n_hidden'] = n_hidden
    csg_net_args['bidirectional'] = bidirectional
    csg_net_args['n_layer'] = n_layer
    csg_net_args['bert_length'] = bert_length

    net = EventAugmentedSumm(configdgm, csg_net_args, batch_size)

    net_args = csg_net_args
    net_args['configdgm'] = configdgm
    net_args['configIDGL'] = configIDGL

    return net, net_args


def configure_training(opt, lr, clip_grad, lr_decay, batch_size, bert):
    """ supports Adam optimizer only"""
    assert opt in ['adam', 'adagrad']
    opt_kwargs = {}
    opt_kwargs['lr'] = lr

    train_params = {}
    if opt == 'adagrad':
        opt_kwargs['initial_accumulator_value'] = 0.1
    train_params['optimizer'] = (opt, opt_kwargs)
    train_params['clip_grad_norm'] = clip_grad
    train_params['batch_size'] = batch_size
    train_params['lr_decay'] = lr_decay
    if bert:
        PAD = 1
    else:
        PAD = 0
    nll = lambda logit, target: F.nll_loss(logit, target, reduce=False)

    def criterion(logits, targets):
        return sequence_loss(logits, targets, nll, pad_idx=PAD)

    print('pad id:', PAD)
    return criterion, train_params


def build_loaders(cuda, debug, bert_model='roberta-base'):
    tokenizer = RobertaTokenizer.from_pretrained(bert_model)

    # coll_fn is needed to filter out too short abstracts (<100) and articles (<300)
    train_loader = DataLoader(
        CDSRDataset('train', args.data_dir), batch_size=args.batch,
        shuffle=not debug,
        num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn
    )

    val_loader = DataLoader(
        CDSRDataset('train', args.data_dir), batch_size=args.batch,
        shuffle=False, num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn
    )

    return train_loader, val_loader, tokenizer


class BasicTrainer:
    """ Basic trainer with minimal function and early stopping"""
    def __init__(self, ckpt_freq, patience, scheduler, cuda, grad_fn, word2id, save_dir):

        self._ckpt_freq = ckpt_freq
        self._patience = patience
        self._save_dir = save_dir

        self._scheduler = scheduler

        self._epoch = 0
        self._cuda = cuda

        self._grad_fn = grad_fn
        self._batchify = compose(
            batchify_fn_copy_bert(word2id, cuda=self._cuda),
            convert_batch_copy_bert(word2id, args.max_art),
            prepro_fn_copy_bert(word2id, args.max_art, args.max_abs)
        )

    def train(self, net, train_loader, val_loader, optimizer):

        while True:
            net.train()
            self._epoch += 1

            for batch in tqdm(train_loader, leave=False):

                fw_args = self._batchify(batch)

                loss = net(*fw_args)

                loss.backward()

                log_dict = {'loss': loss.item()}

                if self._grad_fn is not None:
                    log_dict.update(self._grad_fn())

                optimizer.step()
                net.zero_grad()

            if self._epoch % self._ckpt_freq == 0:
                stop = self.checkpoint()
                if stop:
                    break

    def checkpoint(self):
        # compute loss on validation set
        val_metric = self.validate()

        # save model weights and optimizer
        self._pipeline.checkpoint(
            join(self._save_dir, 'ckpt'), self._epoch, val_metric)
        if isinstance(self._scheduler, ReduceLROnPlateau):
            self._scheduler.step(val_metric)
        else:
            self._scheduler.step()

        # check if the number of times in a row that we don't experience an improvement
        # is greater than patience e.g. 5, if that's the case we interrupt training
        stop = self.check_stop(val_metric)
        return stop

    def check_stop(self, val_metric):
        if self._best_val is None:
            self._best_val = val_metric
        elif val_metric < self._best_val:
            self._current_p = 0
            self._best_val = val_metric
        else:
            self._current_p += 1
        return self._current_p >= self._patience


def main(args):
    import logging
    logging.basicConfig(level=logging.ERROR)

    train_loader, val_loader, tokenizer = build_loaders(args.cuda, args.debug)

    net, net_args = configure_net(args.configdgm, args.configIDGL, len(tokenizer.encoder), args.emb_dim,
                                  args.n_hidden, args.bi, args.n_layer, args.batch, args.max_art)

    criterion, train_params = configure_training(
        'adam', args.lr, args.clip, args.decay, args.batch, args.bert
    )

    # save experiment setting
    if not exists(args.path):
        os.makedirs(args.path)
    with open(join(args.path, 'vocab.pkl'), 'wb') as f:
        pkl.dump(tokenizer.encoder, f, pkl.HIGHEST_PROTOCOL)

    meta = {'net_args': net_args, 'train_args': train_params}
    with open(join(args.path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)

    # prepare trainer
    if args.cuda:
        net = net.cuda()

    val_fn = basic_validate(net)
    grad_fn = get_basic_grad_fn(net, args.clip)

    optimizer = optim.AdamW(net.parameters(), **train_params['optimizer'][1])

    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                  factor=args.decay, min_lr=0,
                                  patience=args.lr_p)

    trainer = BasicTrainer(args.ckpt_freq, args.patience, scheduler, args.cuda, grad_fn, tokenizer, args.path)

    print('start training with the following hyper-parameters:')
    print(meta)
    trainer.train(net, train_loader, val_loader, optimizer)


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    parser = argparse.ArgumentParser(
        description='training of the abstractor (ML)'
    )
    parser.add_argument('--path', required=True, help='root of the model')
    parser.add_argument('--key', type=str, default='extracted_combine', help='constructed sentences')

    parser.add_argument('--vsize', type=int, action='store', default=50000,
                        help='vocabulary size')
    parser.add_argument('--emb_dim', type=int, action='store', default=128,
                        help='the dimension of word embedding')
    parser.add_argument('--w2v', action='store',
                        help='use pretrained word2vec embedding')
    parser.add_argument('--n_hidden', type=int, action='store', default=256,
                        help='the number of hidden units of LSTM')
    parser.add_argument('--n_layer', type=int, action='store', default=1,
                        help='the number of layers of LSTM')

    parser.add_argument('--docgraph', action='store_true', help='uses gat encoder')
    parser.add_argument('--paragraph', action='store_true', help='encode topic flow')
    parser.add_argument('--mask_type', action='store', default='soft', type=str,
                        help='none, encoder, soft')
    parser.add_argument('--graph_layer', type=int, default=1, help='graph layer number')
    parser.add_argument('--adj_type', action='store', default='edge_as_node', type=str,
                        help='concat_triple, edge_up, edge_down, no_edge, edge_as_node')
    parser.add_argument('--gold_key', action='store', default='summary_worthy', type=str,
                        help='attention type')
    parser.add_argument('--feat', action='append', default=['node_freq'])
    parser.add_argument('--bert', action='store_true', help='use bert!')
    parser.add_argument('--bertmodel', action='store', type=str,
                        default='deep_event_mine/data/bert/scibert_scivocab_cased',
                        help='pre-trained model file path')
    parser.add_argument('--configdgm', action='store', default='play.yaml',
                        help='configuration file name for DeepGraphMine e.g. example1.yaml')
    parser.add_argument('--configIDGL', action='store', default='idgl.yml',
                        help='configuration file name for IDGL e.g. example2.yaml')
    parser.add_argument('--data_dir', action='store', default='CDSR_data',
                        help='directory where the data is stored')

    # length limit
    parser.add_argument('--max_art', type=int, action='store', default=1000,
                        help='maximun words in a single article sentence')
    parser.add_argument('--max_abs', type=int, action='store', default=700,
                        help='maximun words in a single abstract sentence')
    # training options
    parser.add_argument('--lr', type=float, action='store', default=1e-3,
                        help='learning rate')
    parser.add_argument('--decay', type=float, action='store', default=0.5,
                        help='learning rate decay ratio')
    parser.add_argument('--lr_p', type=int, action='store', default=0,
                        help='patience for learning rate decay')
    parser.add_argument('--clip', type=float, action='store', default=2.0,
                        help='gradient clipping')
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='the training batch size')
    parser.add_argument('--num_worker', type=int, action='store', default=4,
                        help='cpu num using for dataloader')
    parser.add_argument(
        '--ckpt_freq', type=int, action='store', default=9000,
        help='number of update steps for checkpoint and validation'
    )
    parser.add_argument('--patience', type=int, action='store', default=5,
                        help='patience for early stopping')

    parser.add_argument('--debug', action='store_true',
                        help='run in debugging mode')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    parser.add_argument('--load_from', type=str, default=None,
                        help='disable GPU training')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    args = parser.parse_args()

    args.bi = True
    if args.docgraph or args.paragraph:
        args.gat = True
    else:
        args.gat = False
    if args.paragraph:
        args.topic_flow_model = True
    else:
        args.topic_flow_model = False

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        torch.cuda.set_device(args.gpu_id)

    args.n_gpu = 1

    print(args)
    main(args)
