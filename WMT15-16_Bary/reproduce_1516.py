import os
import json
import math
from tqdm import tqdm
import scipy.stats as stats
import pandas as pd
from matplotlib import rcParams
from collections import defaultdict
import sys
sys.path.insert(0,'..')
import argparse

from moverscore_re import MoverScorer
from bary_score_re import BaryScoreMetric
from bert_score.scorer import BERTScorer

parser = argparse.ArgumentParser()

parser.add_argument("--metric", type=str, help="metric name")
parser.add_argument("--model", type=str, help="model")
parser.add_argument("--dataset", type=int, help="15 or 16")
parser.add_argument("--batch_size", type=int, default=128)
args = parser.parse_args()
print(args)
if args.metric == 'bertscore':
    scorer = BERTScorer(model_type=args.model,
                        batch_size=args.batch_size,
                        nthreads=4,
                        idf=True,
                        device='cuda')
if args.metric == 'moverscore':
    scorer = MoverScorer(model=args.model,
                         batch_size=args.batch_size,
                         nthread=4,
                         idf=True,
                         device='cuda',
                         n_gram=1)
if args.metric == 'baryscore':
    scorer = BaryScoreMetric(model_name=args.model, use_idfs=True, batch_size=args.batch_size)

if args.dataset==16:
    lps = ['cs-en', 'de-en', 'ru-en', 'fi-en', 'ro-en', 'tr-en']

if args.dataset==15:
    lps = ['cs-en', 'de-en', 'ru-en', 'fi-en']

results = defaultdict(lambda : defaultdict(lambda : float))

for i in tqdm(range(len(lps))):
    data_type = lps[i]
    file_path = './{}_{}_formated.json'.format(args.dataset, data_type)

    with open(file_path, 'r') as file:
        all_data = json.load(file)

    refs, sentences, human = [], [], []
    for k, v in tqdm(all_data.items()):
        refs.append(v['references_sentences'])
        sentences.append(v['system']['wmt{}'.format(args.dataset)]['generated_sentence'])
        human.append(all_data[k]['system']['wmt{}'.format(args.dataset)]['scores']['human'])

    if isinstance(scorer, MoverScorer):
        scores = scorer.score(refs, sentences, refs, sentences)
    if isinstance(scorer, BERTScorer):
        scores = scorer.score(sentences, refs, refs)[2]  # 'F1'
    if isinstance(scorer, BaryScoreMetric):
        scorer.prepare_idfs(sentences, refs)
        scores = scorer.score(sentences, refs)

    c = stats.pearsonr(human, scores)[0]
    results[data_type]['pearson'] = "%.6f" % c

    c = stats.spearmanr(human, scores)[0]
    results[data_type]['spearman'] = "%.6f" % c

    c = stats.kendalltau(human, scores)[0]
    results[data_type]['kendall'] = "%.6f" % c

print(args.metric)
df = pd.DataFrame.from_dict(results)
df.to_csv('{}_{}_{}_5'.format(args.dataset, args.metric,args.model))
print(df)




