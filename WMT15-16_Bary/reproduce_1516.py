import os
import json
import math
from tqdm import tqdm
import scipy.stats as stats
import pandas as pd
from matplotlib import rcParams
from collections import defaultdict
import sys
sys.path.append('../')
#from bary_score import BaryScoreMetric

import argparse
import sys
sys.path.insert(0,'..')
from moverscore_re import MoverScorer
from bary_score_1718 import BaryScoreMetric
#from bary_score import BaryScoreMetric
from bert_score.scorer import BERTScorer
import moverscore_re


parser = argparse.ArgumentParser()

parser.add_argument("--metric", type=str, help="metric name")
parser.add_argument("--model", type=str, help="model")
parser.add_argument("--dataset", type=int, help="15 or 16")
args = parser.parse_args()
print(args)
if args.metric == 'bertscore':
    scorer = BERTScorer(model_type=args.model,
                        batch_size=128,
                        nthreads=4,
                        idf=True,
                        device='cuda')
if args.metric == 'moverscore':
    scorer = MoverScorer(model=args.model,
                         batch_size=128,
                         nthread=4,
                         idf=True,
                         device='cuda',
                         n_gram=1)
if args.metric == 'baryscore':
    scorer = BaryScoreMetric(model_name=args.model, use_idfs=True)

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
        scorer.prepare_idfs(refs, sentences)
        scores = scorer.score(refs,sentences)
        '''
        scores = []
        scores = defaultdict(list)

        scorer.prepare_idfs(refs, sentences)

        for k, v in tqdm(all_data.items()):

            scorer.prepare_idfs(refs, sentences)
            ref = v['references_sentences']
            sentence = v['system']['wmt{}'.format(args.dataset)]['generated_sentence']

            
            score = scorer.evaluate_batch(ref, sentence)[0]
            scores.append(score)
            
            scores_s = scorer.evaluate_batch(ref, sentence)
            #print(scores_s)
            for score, value in scores_s.items():
                scores[score].append(value[0])
            #print(scores)
        scores = scores['baryscore_W']
        
        #scores_1 = scorer.score(refs, sentences)
        #scores = [i[0] for i in scores]

        for s1, s2 in zip(scores, scores_1):
            print('{} vs. {}'.format(s1, s2))
            print(s1==s2)
        '''

    c = stats.pearsonr(human, scores)[0]
    results[data_type]['pearson'] = "%.6f" % c

    c = stats.spearmanr(human, scores)[0]
    results[data_type]['spearman'] = "%.6f" % c

    c = stats.kendalltau(human, scores)[0]
    results[data_type]['kendall'] = "%.6f" % c

print(args.metric)
df = pd.DataFrame.from_dict(results)
df.to_csv('{}_{}_{}_ori'.format(args.dataset, args.metric,args.model))
print(df)




