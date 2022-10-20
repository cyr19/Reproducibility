import numpy as np
import pandas as pd
import pickle as pkl
import os
import torch
import sys
sys.path.insert(0, "..")

from tqdm.auto import tqdm
from collections import defaultdict
from moverscore_re import MoverScorer
from bary_score_re import BaryScoreMetric
from bert_score.scorer import BERTScorer
import moverscore_re

wmt18_sys_to_lang_pairs = ['cs-en', 'de-en', 'et-en', 'fi-en', 'ru-en', 'tr-en', 'zh-en']
wmt18_sys_from_lang_pairs = ['en-cs', 'en-de', 'en-et', 'en-fi', 'en-ru', 'en-tr', 'en-zh']
wmt18_sys_lang_pairs = wmt18_sys_to_lang_pairs + wmt18_sys_from_lang_pairs

import argparse


def get_wmt18_seg_data(lang_pair):
    src, tgt = lang_pair.split('-')
    
    RRdata = pd.read_csv(
        "wmt18/wmt18/wmt18-metrics-task-package/manual-evaluation/RR-seglevel.csv", sep=' ')
    # RRdata_lang = RRdata[RRdata['LP'] == lang_pair] # there is a typo in this data. One column name is missing in the header
    RRdata_lang = RRdata[RRdata.index == lang_pair]

    systems = set(RRdata_lang['BETTER'])
    systems.update(list(set(RRdata_lang['WORSE'])))
    systems = list(systems)
    sentences = {}
    for system in systems:
        with open("wmt18/wmt18/wmt18-metrics-task-package/input/wmt18-metrics-task-nohybrids/system-outputs/newstest2018/{}/newstest2018.{}.{}".format(lang_pair, system, lang_pair)) as f:
            sentences[system] = f.read().split("\n")

    with open("wmt18/wmt18/wmt18-metrics-task-package/input/wmt18-metrics-task-nohybrids/"
              "references/{}".format('newstest2018-{}{}-ref.{}'.format(src, tgt, tgt))) as f:
        references = f.read().split("\n")

    ref = []
    cand_better = []
    cand_worse = []
    for index, row in RRdata_lang.iterrows():
        cand_better += [sentences[row['BETTER']][row['SID']-1]]
        cand_worse += [sentences[row['WORSE']][row['SID']-1]]
        ref += [references[row['SID']-1]]

    return ref, cand_better, cand_worse


def kendell_score(scores_better, scores_worse):
    total = len(scores_better)
    if torch.is_tensor(scores_better):
        correct = torch.sum(scores_better > scores_worse).item()
    else:
        #correct = np.sum(scores_better > scores_worse)

        correct = 0
        for i, j in zip(scores_better, scores_worse):
            if i > j:
                correct += 1

    incorrect = total - correct
    return (correct - incorrect)/total


def get_wmt18_seg_bert_score(lang_pair, scorer, cache=False, from_en=True):
    filename = ''
    '''
    if from_en:
        
        if scorer.idf:
            filename = "cache_score/from_en/18/{}/wmt18_seg_from_{}_{}_idf.pkl".format(scorer.model_type, *lang_pair.split('-'))
        else:
            filename = "cache_score/from_en/18/{}/wmt18_seg_from_{}_{}.pkl".format(scorer.model_type, *lang_pair.split('-'))
    else:
        if scorer.idf:
            filename = "cache_score/to_en/18/{}/wmt18_seg_to_{}_{}_idf.pkl".format(scorer.model_type, *lang_pair.split('-'))
        else:
            filename = "cache_score/to_en/18/{}/wmt18_seg_to_{}_{}.pkl".format(scorer.model_type, *lang_pair.split('-'))
        '''
    if os.path.exists(filename):

        with open(filename, "rb") as f:
            return pkl.load(f)

    else:
        refs, cand_better, cand_worse = get_wmt18_seg_data(lang_pair)

        cands = list(set(cand_worse).union(set(cand_better)))

        if isinstance(scorer, BERTScorer):
            scorer.compute_idf(refs)
            scores_better = scorer.score(cand_better, refs)[2]
            scores_worse = scorer.score(cand_worse, refs)[2]

        if isinstance(scorer, BaryScoreMetric):
            scorer.prepare_idfs(refs, cands)
            scores_better = scorer.score(refs, cand_better)
            scores_worse = scorer.score(refs, cand_worse)

        if isinstance(scorer, MoverScorer):
            scorer._idf_dict = moverscore_re.get_idf_dict(cands, scorer._tokenizer), moverscore_re.get_idf_dict(refs, scorer._tokenizer)
            scores_better = scorer.score(refs, cand_better)
            scores_worse = scorer.score(refs, cand_worse)

        if cache:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as f:
                pkl.dump((scores_better, scores_worse), f)
        return scores_better, scores_worse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="wmt18", help="path to wmt16 data")
    #parser.add_argument("-m", "--model", nargs="+", help="models to tune")
    parser.add_argument("--metric", type=str, help="metric name")
    parser.add_argument("-l", "--log_file", default="wmt18_log.csv", help="log file path")
    #parser.add_argument("--idf", action="store_true")
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument(
        "--lang_pairs",
        nargs="+",
        default=wmt18_sys_to_lang_pairs,
        help="language pairs used for tuning",
    )
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    header = 'model_type'
    for lang_pair in args.lang_pairs + ['avg']:
        header += f',{lang_pair}'
    print(header)
    if not os.path.exists(args.log_file):
        with open(args.log_file, 'w') as f:
            print(header, file=f)
    
    #print(args.model)
    #for model_type in args.model:
    if args.metric == 'bertscore':
        scorer = BERTScorer(model_type='bert-base-uncased',
                              batch_size=args.batch_size,
                              nthreads=4,
                              idf=True,
                              device='cuda')
    if args.metric == 'moverscore':
        scorer = MoverScorer(model='bert-base-uncased',
                                   batch_size=args.batch_size,
                                   nthread=4,
                                   idf=True,
                                   device='cuda',
                                   n_gram=1)
    if args.metric == 'baryscore':
        scorer = BaryScoreMetric(use_idfs=True, batch_size=args.batch_size)
    print(args.metric)
    results = defaultdict(dict)
    for lang_pair in tqdm(args.lang_pairs):
        #print(lang_pair)
        scores_better, scores_worse = get_wmt18_seg_bert_score(lang_pair, scorer, cache=False, from_en=False)
        #print(len(scores_better))
        #for sb, sw, name in zip(scores_better, scores_worse, ["P", "R", "F"]):
        #for sb, sw in zip(scores_better, scores_worse):
            #results[lang_pair][f"{model_type} {name}"] = kendell_score(sb, sw)
        results[lang_pair] = kendell_score(scores_better, scores_worse)
    #for name in ["P", "R", "F"]:
    temp = []
    for lang_pair in args.lang_pairs:
        temp.append(results[lang_pair])
    results["avg"] = np.mean(temp)

    msg = f"{args.metric} (idf)"# if args.idf else f"{args.metric}"
    for lang_pair in args.lang_pairs + ['avg']:
        msg += f",{results[lang_pair]}"
    print(msg)
    with open(args.log_file, "a") as f:
        print(msg, file=f)

    del scorer


if __name__ == "__main__":
    main()
