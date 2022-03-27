from wmt_eval import run_wmt
import argparse
import sys
sys.path.insert(0,'..')
from moverscore_re import MoverScorer
from bary_score_1718 import BaryScoreMetric
from bert_score.scorer import BERTScorer
import moverscore_re


parser = argparse.ArgumentParser()

parser.add_argument("--metric", type=str, help="metric name")
parser.add_argument("--model", type=str, help="model")
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
    scorer = BaryScoreMetric(model_name=args.model, use_idfs=True, batch_size=128)

run_wmt(scorer,17)

