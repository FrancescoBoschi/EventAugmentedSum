import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from transformers import T5Model

from graph_augmented_sum.data.batcher import pad_batch_tensorize
from .attention import step_attention, badanau_attention, copy_from_node_attention, hierarchical_attention
from .util import len_mask, sequence_mean, sequence_loss
from .summ import Seq2SeqSumm, AttentionalLSTMDecoder
from . import beam_search as bs
from graph_augmented_sum.utils import PAD, UNK, START, END
from graph_augmented_sum.model.graph_enc import gat_encode, node_mask
from graph_augmented_sum.model.extract import MeanSentEncoder
from graph_augmented_sum.model.rnn import lstm_multiembedding_encoder
from graph_augmented_sum.model.graph_enc import subgraph_encode
from graph_augmented_sum.model.roberta import RobertaEmbedding
from .rnn import MultiLayerLSTMCells

MAX_FREQ = 100
INIT = 1e-2
BERT_MAX_LEN = 512


class _CopyLinear(nn.Module):
    def __init__(self, context_dim, state_dim, input_dim, side_dim1, side_dim2=None, bias=True):
        super().__init__()
        self._v_c = nn.Parameter(torch.Tensor(context_dim))
        self._v_s = nn.Parameter(torch.Tensor(state_dim))
        self._v_i = nn.Parameter(torch.Tensor(input_dim))
        self._v_s1 = nn.Parameter(torch.Tensor(side_dim1))
        init.uniform_(self._v_c, -INIT, INIT)
        init.uniform_(self._v_s, -INIT, INIT)
        init.uniform_(self._v_i, -INIT, INIT)
        init.uniform_(self._v_s1, -INIT, INIT)
        if side_dim2 is not None:
            self._v_s2 = nn.Parameter(torch.Tensor(side_dim2))
            init.uniform_(self._v_s2, -INIT, INIT)
        else:
            self._v_s2 = None

        if bias:
            self._b = nn.Parameter(torch.zeros(1))
        else:
            self._b = None

    def forward(self, context, state, input_, side1, side2=None):
        output = (torch.matmul(context, self._v_c.unsqueeze(1))
                  + torch.matmul(state, self._v_s.unsqueeze(1))
                  + torch.matmul(input_, self._v_i.unsqueeze(1))
                  + torch.matmul(side1, self._v_s1.unsqueeze(1)))
        if side2 is not None and self._v_s2 is not None:
            output += torch.matmul(side2, self._v_s2.unsqueeze(1))
        if self._b is not None:
            output = output + self._b.unsqueeze(0)
        return output


class CopySummIDGL(Seq2SeqSumm):
    def __init__(self, vocab_size, emb_dim,
                 n_hidden, bidirectional, n_layer, side_dim, dropout=0.0, bert=False, bert_length=512):
        super().__init__(vocab_size, emb_dim,
                         n_hidden, bidirectional, n_layer, dropout)
        self._bert = bert
        if self._bert:
            self._bert_model = RobertaEmbedding()
            self._embedding = self._bert_model._embedding
            self._embedding.weight.requires_grad = False
            emb_dim = self._embedding.weight.size(1)
            self._bert_max_length = bert_length
            self._enc_lstm = nn.LSTM(
                emb_dim, n_hidden, n_layer,
                bidirectional=bidirectional, dropout=dropout
            )

            self._projection = nn.Sequential(
                nn.Linear(2 * n_hidden, n_hidden),
                nn.Tanh(),
                nn.Linear(n_hidden, emb_dim, bias=False)
            )
            self._dec_lstm = MultiLayerLSTMCells(
                emb_dim * 2, n_hidden, n_layer, dropout=dropout
            )
            # overlap between 2 lots to avoid breaking paragraph
            self._bert_stride = 256

        self._copy = _CopyLinear(n_hidden, n_hidden, 2 * emb_dim, side_dim, side_dim)

        graph_hsz = n_hidden

        enc_lstm_in_dim = emb_dim
        self._enc_lstm = nn.LSTM(
            enc_lstm_in_dim, n_hidden, n_layer,
            bidirectional=bidirectional, dropout=dropout
        )

        # node attention
        self._attn_s1 = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._attns_wm = nn.Parameter(torch.Tensor(graph_hsz, n_hidden))
        self._attns_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._attns_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._attns_wm)
        init.xavier_normal_(self._attns_wq)
        init.xavier_normal_(self._attn_s1)
        init.uniform_(self._attns_v, -INIT, INIT)
        self._graph_proj = nn.Linear(graph_hsz, graph_hsz)


        self._projection_decoder = nn.Sequential(
            nn.Linear(3 * n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, emb_dim, bias=False)
        )
        self._decoder_supervision = False

        self._decoder = CopyDecoderGAT(
            self._copy, self._attn_s1, self._attns_wm, self._attns_wq, self._attn_v,
            self._attn_copyh, self._attn_copyv, None, None, False,
            self._embedding, self._dec_lstm, self._attn_wq, self._projection_decoder, self._attn_wb, self._attn_v,
        )

    def forward(self, artinfo, absinfo, node_vec, node_num):
        """
        - article: Tensor of size (n_split_docs, 512) where n_split_docs doesn't correspond to the number of documents
                   involved, but to the number of lots of tokens necessary to represent the original articles. E.g. when
                   we have 1 document as input the size is (2, 512), each element represents a token index.

        - art_lens: batch_size-dimensional list containing the number of tokens in each article e.g [745, 631, .....]

        - abstract: tensor of size (batch_size, max_len_abstract) e.g. (32, 51), each token index represents the tokens
                    contained in the abstract

        - extend_art: tensor of size (batch_size, max(art_lens)), for each article we have all the token indexes for each article
                      padded to the maximum number of tokens in an article e.g (32, 745)

        - extend_vsize: highest value for a token index
        """

        article, art_lens, extend_art, extend_vsize = artinfo
        abstract, target = absinfo

        # attention: contains the W_6 * h_k vectors (32, 775, 256)
        # init_dec_states[0]][0]: final hidden state ready for the decoder single-layer unidirectional LSTM
        # init_dec_states[0][1]: final cell state ready for the decoder single-layer unidirectional LSTM
        # init_dec_states[1] it's going do be used together with self._embedding(tok).squeeze(1)
        # as the feature embeddings of the decoder single-layer unidirectional LSTM
        # basically it's the concatenation of the final hidden state and the average of all
        # the token embeddings W_6 * h_k in the article e.g. (32, 768)
        attention, init_dec_states = self.encode(article, art_lens)

        sw_mask = None
        ext_info = None

        mask = len_mask(art_lens, attention.device).unsqueeze(-2)

        # logit: contains all the logits for each prediction that has to be made in the batch e.g. (190, 50265)
        #        where 190 is the number of tokens that have to be predicted in the 3 target documents
        logit, selections = self._decoder(
            (attention, mask, extend_art, extend_vsize),
            abstract,
            node_vec,
            node_num,
            init_dec_states,
            sw_mask,
            ext_info
        )

        nll = lambda logit, target: F.nll_loss(logit, target, reduce=False)
        loss = sequence_loss(logit, target, nll, pad_idx=PAD).mean()

        return loss

    def encode(self, article, art_lens=None):

        # We employ LSTM models with 256-dimensional
        # hidden states for the document encoder (128 each
        # direction) and the decoder
        # size = (2, 32, 256)
        # 2 = n_layer * 2 because bidirectional = True and n_layer = 1
        # 32 is batch_size
        size = (
            self._init_enc_h.size(0),
            len(art_lens) if art_lens else 1,
            self._init_enc_h.size(1)
        )

        # initial encoder states
        # _init_enc_h initial hidden state (2, 32, 256) we have a 2 because we have a bidirectional LSTM
        # _init_enc_c initial cell state (2, 32, 256)
        init_enc_states = (
            self._init_enc_h.unsqueeze(1).expand(*size),
            self._init_enc_c.unsqueeze(1).expand(*size)
        )

        # e.g. self._bert_max_length=1024
        if self._bert_max_length > 512:
            source_nums = art_lens

        # no finetuning Roberta weights
        with torch.no_grad():
            # bert_out[0] (n_split_docs, 512, 768) contains token embeddings
            # bert_out[1] (n_split_docs, 768) contains sentence embeddings
            bert_out = self._bert_model(article)

        # e.g. (n_split_docs, 512, 768) so we obtain an embedding for each token
        # in the 'lots'
        bert_hidden = bert_out[0]
        if self._bert_max_length > 512:
            # e.g. 768
            hsz = bert_hidden.size(2)
            batch_id = 0

            # max(art_lens) e.g 775
            max_source = max(source_nums)

            bert_hiddens = []
            # e.g. 512
            max_len = bert_hidden.size(1)
            for source_num in source_nums:
                # tensor of zeros of size (max(art_lens), 768) e.g. (775, 768)
                source = torch.zeros(max_source, hsz).to(bert_hidden.device)
                if source_num < BERT_MAX_LEN:
                    source[:source_num, :] += bert_hidden[batch_id, :source_num, :]
                    batch_id += 1
                else:
                    # fill the first 512 tokens of the article
                    source[:BERT_MAX_LEN, :] += bert_hidden[batch_id, :BERT_MAX_LEN, :]
                    batch_id += 1
                    start = BERT_MAX_LEN
                    # now we deal with the remaining  source_num - BERT_MAX_LEN tokens e.g. 745 - 212
                    while start < source_num:
                        # print(start, source_num, max_source)
                        if start - self._bert_stride + BERT_MAX_LEN < source_num:
                            end = start - self._bert_stride + BERT_MAX_LEN
                            batch_end = BERT_MAX_LEN
                        else:
                            end = source_num
                            batch_end = source_num - start + self._bert_stride
                        source[start:end, :] += bert_hidden[batch_id, self._bert_stride:batch_end, :]
                        batch_id += 1
                        start += (BERT_MAX_LEN - self._bert_stride)
                bert_hiddens.append(source)

            # now bert hidden has changed size (batch_size, max(art_lens), 768) e.g. (32, 775, 768)
            # so now we have the token embeddings organised for each article
            bert_hidden = torch.stack(bert_hiddens)
        # article = self._bert_relu(self._bert_linear(bert_hidden))
        article = bert_hidden

        # enc_arts (max(art_lens), batch_size, 512) e.g. (775, 32, 512) each vector represents h_k of size 512
        # final_states: tuple of size 2 with each element of size e.g. (2, 32, 256)
        #               final_states[0] contains the final hidden states in both directions that's why we have a 2
        #               final_states[1] contains the final cell states in both directions
        enc_art, final_states = lstm_multiembedding_encoder(
            article, self._enc_lstm, art_lens,
            init_enc_states, None, {}, {}
        )

        if self._enc_lstm.bidirectional:
            h, c = final_states
            # final_states: tuple of size 2 with each element of size e.g. (1, 32, 512)
            # basically we concatenate the final hidden and cell states from both direction
            final_states = (
                torch.cat(h.chunk(2, dim=0), dim=2),
                torch.cat(c.chunk(2, dim=0), dim=2)
            )

        else:
            # in_features=512, out_features=256
            init_h = torch.stack([self._dec_h(s)
                                  for s in final_states[0]], dim=0)
            init_c = torch.stack([self._dec_c(s)
                                  for s in final_states[1]], dim=0)

            # init_dec_states[0]: final hidden state ready for the decoder single-layer unidirectional LSTM
            # init_dec_states[1]: final cell state ready for the decoder single-layer unidirectional LSTM
            init_dec_states = (init_h, init_c)

            # self._attn_wm is of size e.g (512, 256) so we get from (775, 32, 512)
            # to (775, 32, 256) and finally after transposing to (32, 775, 256)
            # basically we perform W_6 * h_k
            attention = torch.matmul(enc_art, self._attn_wm).transpose(0, 1)

            # we write init_h[-1] because we want the last layer output
            # we can have multiple layers, by default we just have 1
            # init_attn_out it's going do be used together with self._embedding(tok).squeeze(1)
            # as the feature embeddings of the decoder single-layer unidirectional LSTM
            # basically it's the concatenation of the final hidden state and the average of all
            # the token embeddings W_6 * h_k in the article e.g. (32, 768)
            init_attn_out = self._projection(torch.cat(
                [init_h[-1], sequence_mean(attention, art_lens, dim=1)], dim=1
            ))
            return attention, (init_dec_states, init_attn_out)

    def encode_bert(self, article, feature_dict, art_lens=None):
        size = (
            self._init_enc_h.size(0),
            len(art_lens) if art_lens else 1,
            self._init_enc_h.size(1)
        )
        init_enc_states = (
            self._init_enc_h.unsqueeze(1).expand(*size),
            self._init_enc_c.unsqueeze(1).expand(*size)
        )
        feature_embeddings = {}

        enc_art, final_states = lstm_multiembedding_encoder(
            article, self._enc_lstm, art_lens,
            init_enc_states, self._embedding,
            feature_embeddings, feature_dict
        )
        if self._enc_lstm.bidirectional:
            h, c = final_states
            final_states = (
                torch.cat(h.chunk(2, dim=0), dim=2),
                torch.cat(c.chunk(2, dim=0), dim=2)
            )
        init_h = torch.stack([self._dec_h(s)
                              for s in final_states[0]], dim=0)
        init_c = torch.stack([self._dec_c(s)
                              for s in final_states[1]], dim=0)
        init_dec_states = (init_h, init_c)
        attention = torch.matmul(enc_art, self._attn_wm).transpose(0, 1)
        init_attn_out = self._projection(torch.cat(
            [init_h[-1], sequence_mean(attention, art_lens, dim=1)], dim=1
        ))
        return attention, (init_dec_states, init_attn_out)

    def encode_general(self, article, art_lens, extend_art, extend_vsize,
                       nodes, nmask, node_num, feature_dict, adjs,
                       go, eos, unk, max_len):
        attention, init_dec_states = self.encode(article, feature_dict, art_lens)

        _nodes, masks = self._encode_graph(attention, nodes, nmask, None,
                                           None, None, adjs, node_num, node_mask=None,
                                           nodefreq=feature_dict['node_freq'])

        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        attention = (attention, mask, extend_art, extend_vsize)

        return attention, init_dec_states, _nodes

    def rl_step(self, article, art_lens, extend_art, extend_vsize,
                _ns, nmask, node_num, feature_dict, adjs,
                go, eos, unk, max_len, attention, init_dec_states, nodes, sample=False):
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings
        tok = torch.LongTensor([go] * batch_size).to(article.device)
        outputs = []
        attns = []
        seqLogProbs = []
        states = init_dec_states
        for i in range(max_len):
            if sample:
                tok, states, attn_score, sampleProb = self._decoder.sample_step(
                    tok, states, attention, nodes, node_num)
                seqLogProbs.append(sampleProb)
            else:
                tok, states, attn_score, node_attn_score = self._decoder.decode_step(
                    tok, states, attention, nodes, node_num)
            if i == 0:
                unfinished = (tok != eos)
            else:
                it = tok * unfinished.type_as(tok)
                unfinished = unfinished * (it != eos)
            attns.append(attn_score.detach())
            if i == 0:
                outputs.append(tok[:, 0].clone())
            else:
                outputs.append(it[:, 0].clone())
            tok.masked_fill_(tok >= vsize, unk)
            if unfinished.data.sum() == 0:
                break
        if sample:
            return outputs, attns, seqLogProbs
        else:
            return outputs, attns

    def ml_step(self, abstract, node_num, attention, init_dec_states, nodes):
        logit, selections = self._decoder(
            attention,
            init_dec_states, abstract,
            nodes,
            node_num,
            None,
            False,
            None
        )
        return logit

    def greedy(self, article, art_lens, extend_art, extend_vsize,
               nodes, nmask, node_num, feature_dict, adjs,
               go, eos, unk, max_len, tar_in):  # deprecated
        """ greedy decode support batching"""
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings
        attention, init_dec_states = self.encode(article, feature_dict, art_lens)

        nodes, masks = self._encode_graph(attention, nodes, nmask, None,
                                          None, None, adjs, node_num, node_mask=None,
                                          nodefreq=feature_dict['node_freq'])

        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        attention = (attention, mask, extend_art, extend_vsize)
        tok = torch.LongTensor([go] * batch_size).to(article.device)

        outputs = []
        attns = []
        states = init_dec_states
        for i in range(max_len):
            tok, states, attn_score, node_attn_score = self._decoder.decode_step(
                tok, states, attention, nodes, node_num)
            # print('greedy tok:', tok.size())
            if i == 0:
                unfinished = (tok != eos)
                # print('greedy tok:', tok)
            else:
                it = tok * unfinished.type_as(tok)
                unfinished = unfinished * (it != eos)
            attns.append(attn_score.detach())
            if i == 0:
                outputs.append(tok[:, 0].clone())
            else:
                outputs.append(it[:, 0].clone())
            tok.masked_fill_(tok >= vsize, unk)
            if unfinished.data.sum() == 0:
                break
        return outputs, attns

    def sample(self, article, art_lens, extend_art, extend_vsize,
               nodes, nmask, node_num, feature_dict, adjs,
               go, eos, unk, max_len, abstract, ml):  # deprecated
        """ greedy decode support batching"""
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings
        attention, init_dec_states = self.encode(article, feature_dict, art_lens)

        nodes, masks = self._encode_graph(attention, nodes, nmask, None,
                                          None, None, adjs, node_num, node_mask=None,
                                          nodefreq=feature_dict['node_freq'])

        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        attention = (attention, mask, extend_art, extend_vsize)
        tok = torch.LongTensor([go] * batch_size).to(article.device)

        outputs = []
        attns = []
        states = ((init_dec_states[0][0].clone(), init_dec_states[0][1].clone()), init_dec_states[1].clone())
        seqLogProbs = []
        for i in range(max_len):
            tok, states, attn_score, sampleProb = self._decoder.sample_step(
                tok, states, attention, nodes, node_num)
            # print('sample tok:', tok)
            if i == 0:
                unfinished = (tok != eos)
            else:
                it = tok * unfinished.type_as(tok)
                unfinished = unfinished * (it != eos)
            attns.append(attn_score.detach())
            if i == 0:
                outputs.append(tok[:, 0].clone())
            else:
                outputs.append(it[:, 0].clone())
            tok = tok.masked_fill(tok >= vsize, unk)
            seqLogProbs.append(sampleProb)
            if unfinished.data.sum() == 0:
                break

        if ml:
            assert abstract is not None
            logit, _ = self._decoder(
                attention,
                init_dec_states, abstract,
                nodes,
                node_num,
                None,
                False,
                None
            )
            return outputs, attns, seqLogProbs, logit

        return outputs, attns, seqLogProbs

    def batch_decode(self, article, art_lens, extend_art, extend_vsize,
                     nodes, nmask, node_num, feature_dict, adjs,
                     go, eos, unk, max_len):
        """ greedy decode support batching"""
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings
        attention, init_dec_states = self.encode(article, art_lens)
        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        attention = (attention, mask, extend_art, extend_vsize)
        tok = torch.LongTensor([go] * batch_size).to(article.device)
        outputs = []
        attns = []
        states = init_dec_states
        for i in range(max_len):
            tok, states, attn_score = self._decoder.decode_step(
                tok, states, attention)
            attns.append(attn_score)
            outputs.append(tok[:, 0].clone())
            tok.masked_fill_(tok >= vsize, unk)
        return outputs, attns

    def decode(self, article, extend_art, extend_vsize, go, eos, unk, max_len):
        vsize = self._embedding.num_embeddings
        attention, init_dec_states = self.encode(article)
        attention = (attention, None, extend_art, extend_vsize)
        tok = torch.LongTensor([go]).to(article.device)
        outputs = []
        attns = []
        states = init_dec_states
        for i in range(max_len):
            tok, states, attn_score = self._decoder.decode_step(
                tok, states, attention)
            if tok[0, 0].item() == eos:
                break
            outputs.append(tok[0, 0].item())
            attns.append(attn_score.squeeze(0))
            if tok[0, 0].item() >= vsize:
                tok[0, 0] = unk
        return outputs, attns

    def batched_beamsearch(self, article, art_lens,
                           extend_art, extend_vsize,
                           ninfo, rinfo, ext_ninfo,
                           go, eos, unk, max_len, beam_size, diverse=1.0, min_len=35):
        (nodes, nmask, node_num, sw_mask, feature_dict) = ninfo
        (relations, rmask, triples, adjs) = rinfo
        if self._gold:
            sw_mask = sw_mask
        else:
            sw_mask = None
        batch_size = len(art_lens)

        # vocabulary size e.g. 50265
        vsize = self._embedding.num_embeddings
        attention, init_dec_states = self.encode(article, feature_dict, art_lens)

        if self._mask_type == 'soft':
            nodes, masks = self._encode_graph(attention, nodes, nmask, relations,
                                              rmask, triples, adjs, node_num, node_mask=None,
                                              nodefreq=feature_dict['node_freq'])
        else:
            nodes = self._encode_graph(attention, nodes, nmask, relations, rmask, triples, adjs, node_num, sw_mask,
                                       nodefreq=feature_dict['node_freq'])

        ext_info = None

        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        all_attention = (attention, mask, extend_art, extend_vsize)
        attention = all_attention

        # h size e.g. (1, 32, 256)
        # prev size e.g. (3, 768)
        (h, c), prev = init_dec_states

        # list of length batch_size e.g. 32 where all the beam search initialisations are stored
        all_beams = [bs.init_beam(go, hists=(h[:, i, :], c[:, i, :], prev[i]))
                     for i in range(batch_size)]

        max_node_num = max(node_num)
        if sw_mask is not None:
            all_nodes = [(nodes[i, :, :], node_num[i], sw_mask[i, :]) for i in range(len(node_num))]
        else:
            all_nodes = [(nodes[i, :, :], node_num[i]) for i in range(len(node_num))]
        finished_beams = [[] for _ in range(batch_size)]
        outputs = [None for _ in range(batch_size)]

        all_tokens_list = []
        for t in range(max_len):

            all_states = []
            toks = []
            # with t = 0 len(beam) = 1, with t > 0 len(beam) = 5
            # at the first step we just have 1 because we consider just the eos token
            # we don't have to store the top 5 most likely hypothesis
            for beam in filter(bool, all_beams):
                token, states = bs.pack_beam(beam, article.device, use_t5=False)
                all_states.append(states)

                toks.append(token)

            token = torch.stack(toks, dim=1)
            # mask tokens that are not in the vocabulary with the unk token
            token.masked_fill_(token >= vsize, unk)

            # states[0][0] contains the hidden states e.g. (1, 1, 32, 256) at t=0 and (1, 5, 32, 256) at t > 0
            # states[0][1] contains the cell states e.g. (1, 1, 32, 256) at t=0 and (1, 5, 32, 256) at t > 0
            # state[1] contains the prev_states e.g. (1, 32, 768) at t=0 and (5, 32, 768) at t > 0
            states = ((torch.stack([h for (h, _), _ in all_states], dim=2),
                       torch.stack([c for (_, c), _ in all_states], dim=2)),
                      torch.stack([prev for _, prev in all_states], dim=1))

            filtered_nodes = torch.stack([all_nodes[i][0] for i, _beam in enumerate(all_beams) if _beam != []], dim=0)
            filtered_node_num = [all_nodes[i][1] for i, _beam in enumerate(all_beams) if _beam != []]
            if sw_mask is not None:
                filtered_sw_mask = torch.stack([all_nodes[i][2] for i, _beam in enumerate(all_beams) if _beam != []],
                                               dim=0)
            else:
                filtered_sw_mask = None

            filtered_ext_info = None

            if t < min_len:
                force_not_stop = True
            else:
                force_not_stop = False

            topk, lp, states, attn_score = self._decoder.topk_step(
                token, states, attention, beam_size, filtered_nodes,
                filtered_node_num,
                max_node_num=max_node_num, side_mask=filtered_sw_mask, force_not_stop=force_not_stop,
                filtered_ext_info=filtered_ext_info, eos=eos)

            batch_i = 0
            for i, (beam, finished) in enumerate(zip(all_beams,
                                                     finished_beams)):
                if not beam:
                    continue

                finished, new_beam = bs.next_search_beam(
                    beam, beam_size, finished, eos,
                    topk[:, batch_i, :], lp[:, batch_i, :],
                    (states[0][0][:, :, batch_i, :],
                     states[0][1][:, :, batch_i, :],
                     states[1][:, batch_i, :]),
                    attn_score[:, batch_i, :],
                    diverse
                )

                batch_i += 1
                if len(finished) >= beam_size:
                    all_beams[i] = []
                    outputs[i] = finished[:beam_size]
                    # exclude finished inputs
                    (attention, mask, extend_art, extend_vsize
                     ) = all_attention
                    masks = [mask[j] for j, o in enumerate(outputs)
                             if o is None]
                    ind = [j for j, o in enumerate(outputs) if o is None]
                    ind = torch.LongTensor(ind).to(attention.device)
                    attention, extend_art = map(
                        lambda v: v.index_select(dim=0, index=ind),
                        [attention, extend_art]
                    )
                    if masks:
                        mask = torch.stack(masks, dim=0)
                    else:
                        mask = None
                    attention = (
                        attention, mask, extend_art, extend_vsize)
                else:
                    all_beams[i] = new_beam
                    finished_beams[i] = finished
            if all(outputs):
                break
        else:
            for i, (o, f, b) in enumerate(zip(outputs,
                                              finished_beams, all_beams)):
                if o is None:
                    outputs[i] = (f + b)[:beam_size]
        return outputs


class CopyDecoderGAT(AttentionalLSTMDecoder):
    def __init__(self, copy, attn_s1, attns_wm, attns_wq, attns_v, attn_copyh=None, attn_copyv=None,
                 para_wm=None, para_v=None, hierarchical_attention=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._copy = copy
        self._attn_s1 = attn_s1
        self._attns_wm = attns_wm
        self._attns_wq = attns_wq
        self._attns_v = attns_v
        self._attn_copyh = attn_copyh
        self._attn_copyv = attn_copyv
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._hierarchical_attn = hierarchical_attention
        if self._hierarchical_attn:
            assert para_wm is not None and para_v is not None
            self._para_wm = para_wm
            self._para_v = para_v

    def __call__(self, attention, target, nodes, node_num, init_states, side_mask=None, output_attn=None, ext_info=None,
                 paras=None):
        # max abstract length in the batch
        max_len = target.size(1)
        states = init_states
        logits = []
        score_ns = []

        # loop over all
        for i in range(max_len):

            tok = target[:, i:i + 1]
            target_embedding_i = None

            # target token index for each document e.g. [0, 12, ....] (32, 1)
            logit, states, _, score_n, out_attns = self._step(tok, attention, nodes, node_num, states,
                                                              side_mask=side_mask, output_attn=True, ext_info=ext_info,
                                                              paras=paras, target_embedding_i=target_embedding_i)
            logits.append(logit)
            score_ns.append(score_n)

        logit = torch.stack(logits, dim=1)
        return logit, score_ns

    def _step(self, tok, attention, nodes, node_num, states, side_mask=None, output_attn=False, ext_info=None,
              paras=None, target_embedding_i=None):
        # Our summary decoder uses a single-layer unidirectional LSTM
        # with a hidden state st at step t

        out_attentions = {}


        # (32, 768)
        # prev_out  basically it's the concatenation of the final hidden state
        # and the average of all the token embeddings W_6 * h_k in the article
        prev_states, prev_out = states

        # self._embedding(tok).squeeze(1) gets the target token embeddings given by Roberta e.g. (32, 768)
        # lstm has size e.g. (32, 1536) where 1536 is given by the concatenation of 2 768-dimensional embeddings
        decoder_in = torch.cat(
            [self._embedding(tok).squeeze(1), prev_out],
            dim=1)

        # decoder_in the input x of the LSTM cell
        # prev_states[0] contains the last hidden state
        # prev_states[1] contains the last cell state
        states = self._lstm(decoder_in, prev_states)

        # we save the last layer state, by default we just have 1 layer so we just
        # get the first and last state
        # state[0] e.g. (1, 32, 256) ---> decoder_out (32, 256)
        # decoder_out stores the 32 s_t hidden states, 1 for each document
        decoder_out = states[0][-1]

        # W_5 * s_t e.g. (32, 256)
        query = torch.mm(decoder_out, self._attn_w)
        attention, attn_mask, extend_src, extend_vsize = attention

        # nodes attention
        if self._hierarchical_attn:
            node_reps, node_length, para_node_aligns = paras
            para_reps = nodes
            node_reps = torch.matmul(node_reps, self._attns_wm)
            para_reps = torch.matmul(para_reps, self._para_wm)
            query_s = torch.mm(decoder_out, self._attns_wq)
            nmask = len_mask(node_length, attention.device).unsqueeze(-2)
            para_length = node_num
        else:
            # W_4 * G^ where each element of each G^ is v_i^ e.g. (32, 45, 256)
            nodes = torch.matmul(nodes, self._attns_wm)

            # W_3 * s_t e.g. (32, 256)
            query_s = torch.mm(decoder_out, self._attns_wq)

            # (32, 1, 45)
            nmask = len_mask(node_num, attention.device).unsqueeze(-2)
        if self._hierarchical_attn:
            side_n, score_n = hierarchical_attention(query_s, node_reps, node_reps, self._attn_v, para_reps,
                                                     self._para_v, para_node_aligns, mem_mask=nmask,
                                                     hierarchical_length=para_length)
        else:
            if side_mask is not None:
                side_n, score_n = badanau_attention(query_s, nodes, nodes, mem_mask=side_mask.unsqueeze(-2),
                                                    v=self._attns_v, sigmoid=False)
            else:

                # side_n is of size(32, 256), each of the 256-dimensional vector represents c_t^v
                # score is of size e.g. (32, 45) each the 45 scalar represents alpha_i_t^v
                side_n, score_n = badanau_attention(query_s, nodes, nodes, mem_mask=nmask, v=self._attns_v,
                                                    sigmoid=False)

        out_attentions['node_attention'] = score_n.detach()

        # W_7 *  c_t^v e.g (3, 256)
        side_n = torch.mm(side_n, self._attn_s1)

        # context is of size(32, 256), each of the 256-dimensional vector represents c_t
        # score is of size e.g. (32, 775) each the 775 scalar represents alpha_k_t
        context, score = badanau_attention(
            query, attention, attention, mem_mask=attn_mask, bias=self._attn_b, v=self._attn_v, side=side_n)
        out_attentions['word_attention'] = score.detach()

        # self._projection applies an hyperplane 768*256,
        # a tanh activation function and another hyperplane 256*768
        # dec_out is of size (32, 768), it's going to be used also as prev_out
        # at the next step
        # torch.cat([decoder_out, context, side_n] = [s_t|c_t|c_t^v]
        dec_out = self._projection(torch.cat([decoder_out, context, side_n], dim=1))
        score_copy = score

        # extend generation prob to extended vocabulary
        # softmax(W_out * [s_t|c_t|c_t^v])
        gen_prob = self._compute_gen_prob(dec_out, extend_vsize)

        # P_copy = σ(W_copy[s_t|c_t|c_t^v|y_t−1]) e.g. (32, 1)
        copy_prob = torch.sigmoid(self._copy(context, decoder_out, decoder_in, side_n))
        # add the copy prob to existing vocab distribution

        lp = torch.log(
            ((-copy_prob + 1) * gen_prob
             ).scatter_add(
                dim=1,
                index=extend_src.expand_as(score_copy),
                src=score_copy * copy_prob
            ) + 1e-10)  # numerical stability for log

        if output_attn:
            return lp, (states, dec_out), score, score_n, out_attentions

        else:
            return lp, (states, dec_out), score, score_n

    def decode_step(self, tok, states, attention, nodes, node_num, side_mask=None, output_attn=False, ext_info=None,
                    paras=None):
        logit, states, score, score_n = self._step(tok, states, attention, nodes, node_num, side_mask, output_attn,
                                                   ext_info, paras)
        out = torch.max(logit, dim=1, keepdim=True)[1]
        return out, states, score, score_n

    def sample_step(self, tok, states, attention, nodes, node_num, side_mask=None, output_attn=False, ext_info=None,
                    paras=None):
        logit, states, score, score_n = self._step(tok, states, attention, nodes, node_num, side_mask, output_attn,
                                                   ext_info, paras)
        # logprob = F.log_softmax(logit, dim=1)
        logprob = logit
        score = torch.exp(logprob)
        # out = torch.multinomial(score, 1).detach()
        out = torch.multinomial(score, 1)
        sampleProb = logprob.gather(1, out)
        return out, states, score, sampleProb

    def topk_step(self, tok, states, attention, k, nodes, node_num, max_node_num, side_mask=None, force_not_stop=False,
                  filtered_ext_info=None, filtered_para_info=None, eos=END):
        """tok:[BB, B], states ([L, BB, B, D]*2, [BB, B, D])"""
        out_attentions = {}

        # h size (1, beam_size, batch_size, 256) e.g. (1, 5, 3, 256) for t > 0
        (h, c), prev_out = states

        # lstm is not beamable
        nl, _, _, d = h.size()

        # beam = 1 at t = 0, 5 else
        beam, batch = tok.size()
        decoder_in = torch.cat(
            [self._embedding(tok), prev_out], dim=-1)

        # (beam * batch, 768) e.g. (15, 768) for t > 0
        lstm_in = decoder_in.contiguous().view(beam * batch, -1)
        prev_states = (h.contiguous().view(nl, -1, d),
                       c.contiguous().view(nl, -1, d))

        # h size is (1, beam_size*batch_size, 256) e.g. (1, 15, 256)
        h, c = self._lstm(lstm_in, prev_states)
        states = (h.contiguous().view(nl, beam, batch, -1),
                  c.contiguous().view(nl, beam, batch, -1))

        decoder_out = states[0][-1]

        query = torch.matmul(decoder_out, self._attn_w)
        attention, attn_mask, extend_src, extend_vsize = attention

        # nodes attention
        if self._hierarchical_attn:
            node_reps, node_length, para_node_aligns, max_subgraph_node_num = filtered_para_info
            para_reps = nodes
            node_reps = torch.matmul(node_reps, self._attns_wm)
            para_reps = torch.matmul(para_reps, self._para_wm)
            query_s = torch.matmul(decoder_out, self._attns_wq)
            nmask = len_mask(node_length, attention.device, max_num=max_subgraph_node_num).unsqueeze(-2)
            para_length = node_num
        else:
            nodes = torch.matmul(nodes, self._attns_wm)
            query_s = torch.matmul(decoder_out, self._attns_wq)
            nmask = len_mask(node_num, attention.device, max_num=max_node_num).unsqueeze(-2)
        if self._hierarchical_attn:
            side_n, score_n = hierarchical_attention(query_s, node_reps, node_reps, self._attn_v, para_reps,
                                                     self._para_v, para_node_aligns, mem_mask=nmask,
                                                     hierarchical_length=para_length, max_para_num=max_node_num)
        else:
            if side_mask is not None:
                side_n, score_n = badanau_attention(query_s, nodes, nodes, mem_mask=side_mask.unsqueeze(-2),
                                                    v=self._attns_v)
            else:
                side_n, score_n = badanau_attention(query_s, nodes, nodes, mem_mask=nmask, v=self._attns_v)

        out_attentions['node_attention'] = score_n.detach()
        side_n = torch.matmul(side_n, self._attn_s1)

        # attention is beamable

        context, score = badanau_attention(
            query, attention, attention, mem_mask=attn_mask, bias=self._attn_b, v=self._attn_v, side=side_n)

        dec_out = self._projection(torch.cat([decoder_out, context, side_n], dim=-1))
        score_copy = score
        # dec_out = self._projection(torch.cat([lstm_out, context, side_n], dim=-1))

        # copy mechanism is not beamable
        gen_prob = self._compute_gen_prob(
            dec_out.contiguous().view(batch * beam, -1), extend_vsize)

        copy_prob = torch.sigmoid(self._copy(context, decoder_out, decoder_in, side_n)).contiguous().view(-1, 1)
        # copy_prob = torch.sigmoid(
        #     self._copy(context, lstm_out, lstm_in_beamable, side_n)
        # ).contiguous().view(-1, 1)
        lp = torch.log(
            ((-copy_prob + 1) * gen_prob
             ).scatter_add(
                dim=1,
                index=extend_src.expand_as(score_copy).contiguous().view(
                    beam * batch, -1),
                src=score_copy.contiguous().view(beam * batch, -1) * copy_prob
            ) + 1e-8).contiguous().view(beam, batch, -1)
        if force_not_stop:
            lp[:, :, eos] = -1e8

        k_lp, k_tok = lp.topk(k=k, dim=-1)

        return k_tok, k_lp, (states, dec_out), score_copy

    def _compute_gen_prob(self, dec_out, extend_vsize, eps=1e-6):

        # what is represented as in the paper W_out * [s_t|c_t|c_t^v]
        logit = torch.mm(dec_out, self._embedding.weight.t())
        bsize, vsize = logit.size()

        # if we have a token with larger index than vsize that is the vocabulary
        # size of Roberta, we fill the logit vectors with values approximately
        # equal to zero 1e-6
        if extend_vsize > vsize:
            ext_logit = torch.Tensor(bsize, extend_vsize - vsize
                                     ).to(logit.device)
            ext_logit.fill_(eps)
            gen_logit = torch.cat([logit, ext_logit], dim=1)
        else:
            gen_logit = logit

        # softmax(W_out * [s_t|c_t|c_t^v])
        gen_prob = F.softmax(gen_logit, dim=-1)
        return gen_prob

    def _compute_copy_activation(self, context, state, input_, score):
        copy = self._copy(context, state, input_) * score
        return copy


