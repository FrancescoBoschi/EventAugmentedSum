import torch

from deep_event_mine.bert.tokenization import BertTokenizer


class ScibertEmbedding:
    def __init__(self, model, vocab_path):
        self._model = model
        self._tokenizer = BertTokenizer(vocab_path, do_lower_case=False)
        if torch.cuda.is_available():
            self._model.cuda()
        self._model.eval()

        print('Bert initialized')
        self._pad_id = self._tokenizer.vocab['[PAD]']
        self._cls_token = self._tokenizer.vocab['[CLS]']
        self._sep_token = self._tokenizer.vocab['[SEP]']
        self._embedding = self._model.embeddings.word_embeddings
        self._embedding.weight.requires_grad = False
        self._eos = self._tokenizer.vocab['[EOS]']

    def __call__(self, input_ids):
        attention_mask = (input_ids != self._pad_id).float()
        return self._model(input_ids, attention_mask=attention_mask)
