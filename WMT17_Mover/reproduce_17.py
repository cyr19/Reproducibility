import tqdm
import pandas as pd
from mosestokenizer import MosesDetokenizer
from mt_utils import load_data, load_metadata, output_MT_correlation
import sys
sys.path.insert(0,'..')
from moverscore_re import MoverScorer, get_idf_dict
from bert_score.scorer import BERTScorer
from bary_score_re import BaryScoreMetric

#from moverscore_v2 import get_idf_dict, word_mover_score, plot_example
from collections import defaultdict
import os

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--metric", type=str, help="metric name")
parser.add_argument("--model", type=str, help="model")

import json

args = parser.parse_args()
params = vars(args)
print(json.dumps(params, indent=2))


#USERHOME = os.path.expanduser(".")
#MOVERSCORE_DIR = os.environ.get('MOVERSCORE', os.path.join(USERHOME, '.moverscore'))
#data_dir = os.path.join(MOVERSCORE_DIR, 'MT')
data_dir = 'WMT17'

reference_list = dict({
        "newstest2017-csen-ref.en": "cs-en",
        "newstest2017-deen-ref.en": "de-en",
        "newstest2017-fien-ref.en": "fi-en",
        "newstest2017-lven-ref.en": "lv-en",
        "newstest2017-ruen-ref.en": "ru-en",
        "newstest2017-tren-ref.en": "tr-en",
        "newstest2017-zhen-ref.en": "zh-en"
        })


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

import datetime
import random
nowTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
randomNum = random.randint(0,100)
if randomNum <= 10:
  randomNum = str(0) + str(randomNum)
uniqueNum = str(nowTime) + str(randomNum)

file = '{}_{}_{}'.format(args.metric, args.model, uniqueNum)

print(file)

data = []
for _ in reference_list.items():
    reference_path, lp = _
    references = load_data(os.path.join(data_dir, reference_path))

    with MosesDetokenizer('en') as detokenize:
        references = [detokenize(ref.split(' ')) for ref in references]

    all_meta_data = load_metadata(os.path.join(data_dir, lp))
    for i in tqdm.tqdm(range(len(all_meta_data))):
        path, testset, lp, system = all_meta_data[i]
        translations = load_data(path)

        with MosesDetokenizer('en') as detokenize:
            translations = [detokenize(hyp.split(' ')) for hyp in translations]

        if isinstance(scorer, MoverScorer):
            scores = scorer.score(references, translations, references, translations)
        if isinstance(scorer, BERTScorer):
            scores = scorer.score(translations, references, references)[2]  # 'F1'
        if isinstance(scorer, BaryScoreMetric):
            a = scorer.prepare_idfs(translations, references)
            scores = []
            for t, r in zip(translations, references):
                score = scorer.evaluate_batch(t, r)[0]
                scores.append(score)

        df_system = pd.DataFrame(columns=('metric', 'lp', 'testset', 'system', 'sid', 'score'))
        #scores = word_mover_score(references, translations, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=True,
                                  #    batch_size=64)



        num_samples = len(references)
        df_system = pd.DataFrame({'metric': [file] * num_samples,
                               'lp': [lp] * num_samples,
                               'testset': [testset] * num_samples,
                               'system': [system] * num_samples,
                               'sid': [_ for _ in range(1, num_samples + 1)],
                               'score': scores,
                             })
        data.append(df_system) 

results = pd.concat(data, ignore_index=True)
results.to_csv(file + '.seg.score', sep='\t', index=False, header=False)
output_MT_correlation(lp_set=list(reference_list.values()), eval_metric=file)


