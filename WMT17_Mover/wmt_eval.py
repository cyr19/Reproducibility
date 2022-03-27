import pandas as pd

import tqdm
import os

import truecase
from mosestokenizer import MosesDetokenizer
from mt_utils import (find_corpus,
                      find_corpus_foreign,
                      load_data,
                      load_metadata,
                      print_sys_level_correlation,
                      print_seg_level_correlation,
                      df_append,
                      output_mt_correlation_17)

import sys
sys.path.insert(0, "..")

from moverscore_re import MoverScorer
from bert_score.scorer import BERTScorer
from bary_score_1718 import BaryScoreMetric

class wmt_processor():
    def __init__(
            self,
            scorer=None,
            dataset=None,
            from_en=False,

            lp=None
    ):
        self._scorer = scorer

        self.lp = lp
        self.from_en = from_en
        self.dataset = dataset
        if isinstance(scorer, BERTScorer):
            self.metric = 'bertscore'
        if isinstance(scorer, MoverScorer):
            self.metric = 'moverscore'
        if isinstance(scorer, BaryScoreMetric):
            self.metric = 'baryscore'

    def eval(self):
        moverscores = []

        if self.from_en:
            dataset = find_corpus_foreign(self.dataset)
            if self.lp is not None:
                dataset = {k: v for k, v in dataset.items() if v in self.lp}
        else:
            dataset = find_corpus(self.dataset)
            if self.lp is not None:
                dataset = {k: v for k, v in dataset.items() if v in self.lp}

        for pair in dataset.items():
            reference_path, lp = pair

            src, tgt = lp.split('-')
            references = load_data(os.path.join('Dataset/' + self.dataset, 'references', reference_path))
            all_meta_data = load_metadata(os.path.join('Dataset/' + self.dataset, 'system-outputs', lp))


            with MosesDetokenizer(tgt) as detokenize:
                references = [detokenize(s.split(' ')) for s in references]

            for i in tqdm.tqdm(range(len(all_meta_data)), desc=lp):
                path, testset, lp, system = all_meta_data[i]

                translations = load_data(path)
                num_samples = len(references)
                # df_system = pd.DataFrame(columns=('metric', 'lp', 'testset', 'system', 'sid', 'score'))

                with MosesDetokenizer(tgt) as detokenize:
                    translations = [detokenize(s.split(' ')) for s in translations]

                if isinstance(self._scorer, MoverScorer):
                    scores = self._scorer.score(references, translations, references, translations)
                if isinstance(self._scorer, BERTScorer):
                    scores = self._scorer.score(translations, references, references)[2]  # 'F1'
                if isinstance(self._scorer, BaryScoreMetric):
                    self._scorer.prepare_idfs(references, translations)
                    scores = self._scorer.score(references, translations)

                moverscores.append(df_append(self.metric, num_samples, lp, testset, system, scores))

        if self.dataset in ['WMT18', 'WMT19']:
            print_seg_level_correlation('tmp', self.metric, moverscores, list(dataset.values()),
                                        os.path.join('Dataset/' + self.dataset, 'RR-seglevel.csv'))
        elif self.dataset == 'WMT15' or self.dataset == 'WMT16':
                results = pd.concat(moverscores, ignore_index=True)
                results.to_csv('wmt{}/{}+idf-{}.seg.score'.format(self.dataset[-2:],self.metric, self._scorer.idf), sep='\t', index=False, header=False)

        else:
            results = pd.concat(moverscores, ignore_index=True)
            results.to_csv('tmp.seg.score-backup', sep='\t', index=False, header=False)

            if self.from_en:
                dataset_1 = {k: v for k, v in dataset.items() if v == 'en-zh' or v == 'en-ru'}
                dataset_2 = {k: v for k, v in dataset.items() if v != 'en-zh' and v != 'en-ru'}
                if (len(dataset_1) != 0):
                    print('SEG-DA:')

                    output_mt_correlation_17(dataset_1.values(), self.metric, 'tmp.seg.score-backup',
                                             f="Dataset/WMT17/DA-seglevel.csv",
                                             corr='pearson')
                if (len(dataset_2) != 0):
                    print('SEG-RR:')
                    print_seg_level_correlation('tmp', self.metric, moverscores, list(dataset_2.values()),
                                                os.path.join('Dataset/' + self.dataset, 'RR-seglevel.csv'))
            else:
                print('SEG-DA:')
                output_mt_correlation_17(dataset.values(), self.metric, 'tmp.seg.score-backup',
                                         f="Dataset/WMT17/DA-seglevel.csv",
                                         corr='pearson')
        if self.dataset not in ['WMT15', 'WMT16']:
            print_sys_level_correlation('tmp', self.metric, moverscores, list(dataset.values()),
                                        os.path.join('Dataset/' + self.dataset, 'DA-syslevel.csv'), 'pearson')


def run_wmt(scorer, year, from_en=False, lp=None):
    if lp is None:
        print('WMT{}-all'.format(year))
    else:
        print('WMT{}-{}'.format(year, lp))

    wmt = wmt_processor(scorer=scorer,
                        dataset='WMT{}'.format(year),
                        from_en=from_en,
                        lp=lp
                        )
    wmt.eval()
