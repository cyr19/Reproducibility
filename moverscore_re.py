from __future__ import absolute_import, division, print_function

import json

import numpy as np
import torch
import string
from pyemd import emd
from torch import nn
from math import log
from itertools import chain

from pytorch_pretrained_bert import BertTokenizer, BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial

import os
import sys
import requests
import zipfile
import pandas as pd

class MoverScorer:
    """
    MoverScore Scorer Object.
    """

    def __init__(
            self,
            model = 'nli',
            batch_size = 128,
            nthread = 4,
            idf = False,
            idf_sents = None,
            stopwords = None,
            device = None,
            wordpiece = 1,
            ave_wordpieces = False,
            remove_punctuation = True,
            n_gram = 1,
            idf_dict = None
    ):

        assert model is not None
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if model == 'nli':
            USERHOME = os.path.expanduser(".")
            MOVERSCORE_DIR = os.environ.get('MOVERSCORE', os.path.join(USERHOME, '.moverscore'))

            MNLI_BERT = 'https://github.com/AIPHES/emnlp19-moverscore/releases/download/0.6/MNLI_BERT.zip'
            model_dir = os.path.join(MOVERSCORE_DIR)

            def download_MNLI_BERT(url, filename):
                with open(filename, 'wb') as f:
                    response = requests.get(url, stream=True)
                    total = response.headers.get('content-length')

                    if total is None:
                        f.write(response.content)
                    else:
                        downloaded = 0
                        total = int(total)
                        for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                            downloaded += len(data)
                            f.write(data)
                            done = int(50 * downloaded / total)
                            sys.stdout.write('\r[{}{}]'.format('-' * done, '.' * (50 - done)))
                            sys.stdout.flush()
                sys.stdout.write('\n')

            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)

            tarball = os.path.join(model_dir, os.path.basename(MNLI_BERT))
            rawdir = os.path.join(model_dir, 'raw')

            if not os.path.exists(tarball):
                print("Downloading %s to %s" % (MNLI_BERT, tarball))
                download_MNLI_BERT(MNLI_BERT, tarball)

                if tarball.endswith('.zip'):
                    z = zipfile.ZipFile(tarball, 'r')
                    #        z.printdir()
                    z.extractall(model_dir)
                    z.close()
        else:
            model_dir = model

        class BertForSequenceClassification(BertPreTrainedModel):
            def __init__(self, config, num_labels):
                super(BertForSequenceClassification, self).__init__(config)
                self.num_labels = num_labels
                self.bert = BertModel(config)
                self.dropout = nn.Dropout(config.hidden_dropout_prob)
                self.classifier = nn.Linear(config.hidden_size, num_labels)
                self.apply(self.init_bert_weights)

            def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=None):
                encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                          output_all_encoded_layers=True)
                return encoded_layers, pooled_output

        self.batch_size = batch_size
        self.remove_punctuation = remove_punctuation
        self.wordpiece = wordpiece
        self.nthread = nthread
        self.ave_wordpieces = ave_wordpieces
        self.n_gram = n_gram
        self.idf_sents = idf_sents

        #self._tokenizer = BertTokenizer.from_pretrained('./'+model_dir, do_lower_case=True)
        self._tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=True, cache_dir='.cache')
        self._tokenizer.max_len = 512
        #self._tokenizer.return_offsets_mapping=True

        #self._model = BertForSequenceClassification.from_pretrained('./'+model_dir, 3)
        self._model = BertForSequenceClassification.from_pretrained(model_dir, 3, cache_dir='.cache')
        self._model.eval()
        self._model.to(self.device)

        self.idf = idf

        '''
        self._idf_dict = idf_dict
        if self._idf_dict is None and idf_sents is not None and idf:
            self._idf_dict = get_idf_dict(idf_sents, self._tokenizer, nthread)
        '''

        if idf:
            self._idf_dict = None
            if idf_dict is not None:
                self._idf_dict = idf_dict

            if self.idf_sents is not None and idf_dict is None:
                self._idf_dict = get_idf_dict(idf_sents, self._tokenizer, nthread)

        else:
            self._idf_dict = None


        if stopwords is not None:
            self._stopwords = stopwords
        else:
            self._stopwords = []


    def score(self, refs, hyps, ref_idf_sents=None, hyp_idf_sents=None):
        '''
        if self.idf:
            if self.idf_sents is None:
                assert ref_idf_sents is not None and hyp_idf_sents is not None
                ref_idf_dict = get_idf_dict(ref_idf_sents, self._tokenizer, self.nthread)
                hyp_idf_dict = get_idf_dict(hyp_idf_sents, self._tokenizer, self.nthread)
            else:
                idf_dict = get_idf_dict(self.idf_sents, self._tokenizer, self.nthread)
                ref_idf_dict = idf_dict
                hyp_idf_dict = idf_dict
        else:
            ref_idf_dict = defaultdict(lambda : 1.0)
            hyp_idf_dict = defaultdict(lambda : 1.0)
        '''
        sw = []
        if self.idf:
            if self._idf_dict is None:
                assert ref_idf_sents is not None and hyp_idf_sents is not None
                ref_idf_dict = get_idf_dict(ref_idf_sents, self._tokenizer, self.nthread)
                hyp_idf_dict = get_idf_dict(hyp_idf_sents, self._tokenizer, self.nthread)
            else:

                if isinstance(self._idf_dict, defaultdict):

                    ref_idf_dict = self._idf_dict
                    hyp_idf_dict = self._idf_dict
                else:
                    #print('should be here!!!')
                    hyp_idf_dict, ref_idf_dict = self._idf_dict
        else:
            ref_idf_dict = defaultdict(lambda : 1.0)
            hyp_idf_dict = defaultdict(lambda : 1.0)

        #print(len(ref_idf_dict))

        preds = []

        #print(self.device)
        for batch_start in range(0, len(refs), self.batch_size):
            batch_refs = refs[batch_start:batch_start + self.batch_size]
            batch_hyps = hyps[batch_start:batch_start + self.batch_size]

            ref_embedding, ref_lens, ref_masks, ref_idf, ref_tokens = get_bert_embedding(batch_refs, self._model,
                                                                                         self._tokenizer, ref_idf_dict,
                                                                                         device=self.device)
            hyp_embedding, hyp_lens, hyp_masks, hyp_idf, hyp_tokens = get_bert_embedding(batch_hyps, self._model,
                                                                                         self._tokenizer, hyp_idf_dict,
                                                                  device=self.device)

            ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
            hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

            ref_embedding_max, _ = torch.max(ref_embedding[-5:], dim=0, out=None)
            hyp_embedding_max, _ = torch.max(hyp_embedding[-5:], dim=0, out=None)

            ref_embedding_min, _ = torch.min(ref_embedding[-5:], dim=0, out=None)
            hyp_embedding_min, _ = torch.min(hyp_embedding[-5:], dim=0, out=None)

            ref_embedding_avg = ref_embedding[-5:].mean(0)
            hyp_embedding_avg = hyp_embedding[-5:].mean(0)

            ref_embedding = torch.cat([ref_embedding_min, ref_embedding_avg, ref_embedding_max], -1)
            hyp_embedding = torch.cat([hyp_embedding_min, hyp_embedding_avg, hyp_embedding_max], -1)

            for i in range(len(ref_tokens)):

                if not self.ave_wordpieces:
                    if self.wordpiece == 1 and self.remove_punctuation:

                        ref_ids = [k for k, w in enumerate(ref_tokens[i]) if
                                   w not in set(string.punctuation) and '##' not in w and w not in self._stopwords]
                        hyp_ids = [k for k, w in enumerate(hyp_tokens[i]) if
                                   w not in set(string.punctuation) and '##' not in w and w not in self._stopwords]
                        '''
                        tmp = [w for w in set(ref_tokens[i]+hyp_tokens[i]) if w in self._stopwords and '##' not in w and w not in set(string.punctuation)]
                        sw.extend(tmp)
                        '''
                    elif self.wordpiece == 1 and self.remove_punctuation is False:

                        ref_ids = [k for k, w in enumerate(ref_tokens[i]) if '##' not in w and w not in self._stopwords]
                        hyp_ids = [k for k, w in enumerate(hyp_tokens[i]) if '##' not in w and w not in self._stopwords]

                    elif self.wordpiece != 1 and self.remove_punctuation:

                        ref_ids = [k for k, w in enumerate(ref_tokens[i]) if

                                   w not in set(string.punctuation) and not is_stopword(w, self._stopwords)]

                        hyp_ids = [k for k, w in enumerate(hyp_tokens[i]) if

                                   w not in set(string.punctuation) and not is_stopword(w, self._stopwords)]



                    elif self.wordpiece != 1 and self.remove_punctuation is False:


                        ref_ids = [k for k, w in enumerate(ref_tokens[i]) if not is_stopword(w, self._stopwords)]

                        hyp_ids = [k for k, w in enumerate(hyp_tokens[i]) if not is_stopword(w, self._stopwords)]


                    ref_embedding_i, ref_idf_i = load_ngram(ref_ids, ref_embedding[i], ref_idf[i], self.n_gram, 1, device=self.device)
                    hyp_embedding_i, hyp_idf_i = load_ngram(hyp_ids, hyp_embedding[i], hyp_idf[i], self.n_gram, 1, device=self.device)

                else:
                    if self.remove_punctuation:
                        ref_em, ref_i = get_ave_tem_remove(ref_tokens[i], ref_embedding[i], ref_idf[i], self.device, self._stopwords)
                        hyp_em, hyp_i = get_ave_tem_remove(hyp_tokens[i], hyp_embedding[i], hyp_idf[i], self.device, self._stopwords)
                    else:
                        ref_em, ref_i = get_ave_tem_keep(ref_tokens[i], ref_embedding[i], ref_idf[i], self.device, self._stopwords)
                        hyp_em, hyp_i = get_ave_tem_keep(hyp_tokens[i], hyp_embedding[i], hyp_idf[i], self.device, self._stopwords)
                    ref_embedding_i, ref_idf_i = load_ngram(np.arange(len(ref_i)), ref_em, ref_i, self.n_gram, 1,
                                                                device=self.device)
                    hyp_embedding_i, hyp_idf_i = load_ngram(np.arange(len(hyp_i)), hyp_em, hyp_i, self.n_gram, 1,
                                                                device=self.device)


                raw = torch.cat([ref_embedding_i, hyp_embedding_i], 0)
                raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 0.000001)

                distance_matrix = pairwise_distances(raw, raw)

                c1 = np.zeros(len(ref_idf_i) + len(hyp_idf_i), dtype=np.double)
                c2 = np.zeros(len(ref_idf_i) + len(hyp_idf_i), dtype=np.double)

                c1[:len(ref_idf_i)] = ref_idf_i
                c2[-len(hyp_idf_i):] = hyp_idf_i

                c1 = _safe_divide(c1, np.sum(c1))
                c2 = _safe_divide(c2, np.sum(c2))
                score = 1 - emd(c1, c2, distance_matrix.double().cpu().numpy())
                preds.append(score)
        '''
        if self._stopwords is not None:
            print('{} - true stopwords size: {}'.format(len(self._stopwords), len(sw)))
            with open('Dataset/stopwords_lists/true_stopwords/{}.txt'.format(len(self._stopwords)), 'a') as f:
                f.write('\nnew_system\n')
                for w in sw:
                    f.write(w+'\n')
        '''
        return preds


#####################---utils---#####################
def is_stopword(w, stopwords):

    if '##' in w:
        c = w[2:] in stopwords

    else:
        c = w in stopwords

    return c

def get_ave_tem_keep(tokens,embs,idfs,device,stopwords):

    raw_len = len(embs[0])

    new_em, new_idf = torch.reshape(embs[0],(1,raw_len)), [idfs[0]]

    tmp_em = None

    for (k, w), em, idf in zip(enumerate(tokens), embs, idfs):

        if w == '[CLS]' or w == '[SEP]':
            continue
        #print(w)

        if w.startswith('##'):
            tmp_em = torch.cat((tmp_em, torch.reshape(em,(1,raw_len))), 0)
            tmp_idf.append(idf)
            tmp_tokens.append(w[2:])

            if k == len(tokens) - 2:


                if ''.join(tmp_tokens) not in stopwords:

                    new_em = torch.cat((new_em, torch.reshape(torch.mean(tmp_em,0),(1,raw_len))), 0)
                    new_idf.append(np.mean(tmp_idf))

        else:


            if tmp_em is not None:


                if ''.join(tmp_tokens) not in stopwords:
                    new_em = torch.cat((new_em, torch.reshape(torch.mean(tmp_em,0), (1,raw_len))), 0)
                    new_idf.append(np.mean(tmp_idf))

            tmp_em = torch.reshape(em, (1, raw_len))
            tmp_idf = [idf]
            tmp_tokens = [w]

            if k == len(tokens) - 2:

                if ''.join(tmp_tokens) not in stopwords:
                    new_em = torch.cat((new_em, torch.reshape(torch.mean(tmp_em,0),(1,raw_len))), 0)
                    new_idf.append(np.mean(tmp_idf))

    new_em = torch.cat((new_em, torch.reshape(embs[-1], (1,raw_len))), 0).to(device=device)

    new_idf.append(idfs[-1])

    new_idf = torch.tensor(new_idf).to(device=device)

    return new_em, new_idf


def get_ave_tem_remove(tokens, embs, idfs, device, stopwords):
    raw_len = len(embs[0])

    new_em, new_idf = torch.reshape(embs[0], (1, raw_len)), [idfs[0]]

    tmp_em = None

    for (k, w), em, idf in zip(enumerate(tokens), embs, idfs):

        if w == '[CLS]' or w == '[SEP]' or w in set(string.punctuation):
            continue
        # print(w)
        if w.startswith('##'):
            tmp_em = torch.cat((tmp_em, torch.reshape(em, (1, raw_len))), 0)
            tmp_idf.append(idf)
            tmp_tokens.append(w[2:])

            if k == len(tokens) - 2:
                if ''.join(tmp_tokens) not in stopwords:
                    new_em = torch.cat((new_em, torch.reshape(torch.mean(tmp_em, 0), (1, raw_len))), 0)
                    new_idf.append(np.mean(tmp_idf))

        else:

            if tmp_em is not None:
                if ''.join(tmp_tokens) not in stopwords:
                    new_em = torch.cat((new_em, torch.reshape(torch.mean(tmp_em, 0), (1, raw_len))), 0)
                    new_idf.append(np.mean(tmp_idf))

            tmp_em = torch.reshape(em, (1, raw_len))
            tmp_idf = [idf]
            tmp_tokens = [w]

            if k == len(tokens) - 2:
                if ''.join(tmp_tokens) not in stopwords:
                    new_em = torch.cat((new_em, torch.reshape(torch.mean(tmp_em, 0), (1, raw_len))), 0)
                    new_idf.append(np.mean(tmp_idf))

    new_em = torch.cat((new_em, torch.reshape(embs[-1], (1, raw_len))), 0).to(device=device)

    new_idf.append(idfs[-1])

    new_idf = torch.tensor(new_idf).to(device=device)

    return new_em, new_idf

def truncate(tokens,tokenizer):
    if len(tokens) > tokenizer.max_len - 2:
        tokens = tokens[0:(tokenizer.max_len - 2)]
    return tokens

def process(a,tokenizer):
    a = ["[CLS]"]+truncate(tokenizer.tokenize(a),tokenizer)+["[SEP]"]

    a = tokenizer.convert_tokens_to_ids(a)
    return set(a)


def get_idf_dict(arr, tokenizer, nthreads=4):
    idf_count = Counter()
    num_docs = len(arr)

    process_partial = partial(process, tokenizer=tokenizer)

    with Pool(nthreads) as p:
        idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

    idf_dict = defaultdict(lambda : log((num_docs+1)/(1)))
    idf_dict.update({idx:log((num_docs+1)/(c+1)) for (idx, c) in idf_count.items()})
    return idf_dict

def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, :lens[i]] = 1
    return padded, lens, mask

def bert_encode(model, x, attention_mask):
    model.eval()
    x_seg = torch.zeros_like(x, dtype=torch.long)
    with torch.no_grad():
        x_encoded_layers, pooled_output = model(x, x_seg, attention_mask=attention_mask, output_all_encoded_layers=True)
    return x_encoded_layers

def collate_idf(arr, tokenize, tokenizer, numericalize, idf_dict,
                pad="[PAD]", device='cpu'):
    tokens = [["[CLS]"]+truncate(tokenize(a), tokenizer)+["[SEP]"] for a in arr]
    arr = [numericalize(a) for a in tokens]

    idf_weights = [[idf_dict[i] for i in a] for a in arr]

    pad_token = numericalize([pad])[0]

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, pad_token, dtype=torch.float)

    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, padded_idf, lens, mask, tokens

def get_bert_embedding(all_sens, model, tokenizer, idf_dict,
                       batch_size=-1, device='cpu'):

    padded_sens, padded_idf, lens, mask, tokens = collate_idf(all_sens,
                                                      tokenizer.tokenize, tokenizer, tokenizer.convert_tokens_to_ids,
                                                      idf_dict,
                                                      device=device)

    if batch_size == -1: batch_size = len(all_sens)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_sens), batch_size):
            batch_embedding = bert_encode(model, padded_sens[i:i+batch_size],
                                          attention_mask=mask[i:i+batch_size])
            batch_embedding = torch.stack(batch_embedding)
            embeddings.append(batch_embedding)
            del batch_embedding

    total_embedding = torch.cat(embeddings, dim=-3)
    return total_embedding, lens, mask, padded_idf, tokens


def pairwise_distances(x, y=None):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    y_t = torch.transpose(y, 0, 1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def slide_window(a, w=3, o=2):
    if a.size - w + 1 <= 0:
        w = a.size
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::o]
    return view.copy().tolist()


def _safe_divide(numerator, denominator):
    return numerator / (denominator + 0.00001)


def load_ngram(ids, embedding, idf, n, o, device):
    new_a = []
    new_idf = []

    slide_wins = slide_window(np.array(ids), w=n, o=o)
    for slide_win in slide_wins:
        new_idf.append(idf[slide_win].sum().item())
        scale = _safe_divide(idf[slide_win], idf[slide_win].sum(0)).unsqueeze(-1).to(device)
        tmp = (scale * embedding[slide_win]).sum(0)
        new_a.append(tmp)
    new_a = torch.stack(new_a, 0).to(device)
    return new_a, new_idf
