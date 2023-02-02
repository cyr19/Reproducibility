# Reproducibility Issues for BERT-based Evaluation Metrics

This repo contains the main code and data for our EMNLP 2022 paper [Reproducibility Issues for BERT-based Evaluation Metrics]().


## Citation:
```angular2html
@inproceedings{chen-etal-2022-reproducibility,
    title = "Reproducibility Issues for {BERT}-based Evaluation Metrics",
    author = "Chen, Yanran  and
      Belouadi, Jonas  and
      Eger, Steffen",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.192",
    pages = "2965--2989",
    abstract = "Reproducibility is of utmost concern in machine learning and natural language processing (NLP). In the field of natural language generation (especially machine translation), the seminal paper of Post (2018) has pointed out problems of reproducibility of the dominant metric, BLEU, at the time of publication. Nowadays, BERT-based evaluation metrics considerably outperform BLEU. In this paper, we ask whether results and claims from four recent BERT-based metrics can be reproduced. We find that reproduction of claims and results often fails because of (i) heavy undocumented preprocessing involved in the metrics, (ii) missing code and (iii) reporting weaker results for the baseline metrics. (iv) In one case, the problem stems from correlating not to human scores but to a wrong column in the csv file, inflating scores by 5 points. Motivated by the impact of preprocessing, we then conduct a second study where we examine its effects more closely (for one of the metrics). We find that preprocessing can have large effects, especially for highly inflectional languages. In this case, the effect of preprocessing may be larger than the effect of the aggregation mechanism (e.g., greedy alignment vs. Word Mover Distance).",
}
```
> **Abstract**: 
> Reproducibility is of utmost concern in machine learning and natural language processing (NLP). In the field of natural language generation (especially machine translation), the seminal paper of [Post (2018)](https://aclanthology.org/W18-6319/) has pointed out problems of reproducibility of the dominant metric, BLEU, at the time of publication. Nowadays, BERT-based evaluation metrics considerably outperform BLEU. In this paper, we ask whether results and claims from four recent BERT-based metrics can be reproduced. We find that reproduction of claims and results often fails because of (i) heavy undocumented preprocessing involved in the metrics, (ii) missing code and (iii) reporting weaker results for the baseline metrics. (iv) In one case, the problem stems from correlating not to human scores but to a wrong column in the csv file, inflating scores by 5 points. Motivated by the impact of preprocessing, we then conduct a second study where we examine its effects more closely (for one of the metrics). We find that preprocessing can have large effects, especially for highly inflectional languages. In this case, the effect of preprocessing may be larger than the effect of the aggregation mechanism (e.g., greedy alignment vs.\ Word Mover Distance). 

Contact persons: Yanran Chen ([yanran.chen@stud.tu-darmstadt.de](mailto:yanran.chen@stud.tu-darmstadt.de)), Steffen Eger ([steffen.eger@uni-bielefeld.de](mailto:steffen.eger@uni-bielefeld.de))

If you have any questions, don’t hesitate to drop me an email!

## Reproduction on MT
### Reproduction on WMT18
This folder contains the reproduction resources from [Zhang et al., 2019](https://arxiv.org/abs/1904.09675):

https://github.com/Tiiiger/bert_score/tree/master/reproduce
```
cd WMT18_BERT
```
Download WMT18 dataset
```
sh download_wmt18.sh
```


Reproduce the results for WMT18
```
python reproduce_18.py  --metric baryscore
python reproduce_18.py  --metric moverscore
python reproduce_18.py   --metric bertscore
```

### Reproduction on WMT17
This folder contains the reproduction resources from [Zhao et al., 2019](https://arxiv.org/pdf/1909.02622.pdf):

https://github.com/AIPHES/emnlp19-moverscore/tree/master/examples
```
cd WMT17_Mover
```
Reproduce the results for WMT17
```
python reproduce_17.py  --metric baryscore --model bert-base-uncased
python reproduce_17.py   --metric bertscore --model bert-base-uncased
python reproduce_17.py  --metric moverscore --model bert-base-uncased
python reproduce_17.py  --metric moverscore --model nli 
```

### Reproduction on WMT15-16

This folder contains the reproduction resources from [Colombo et al., 2021](https://arxiv.org/abs/2108.12463):

https://github.com/PierreColombo/nlg_eval_via_simi_measures/tree/main/raw_score
```
cd WMT15-16_Bary
```
Download NLI model released in [Zhao et al., 2019](https://arxiv.org/pdf/1909.02622.pdf) for BaryScore. 

```
wget https://github.com/AIPHES/emnlp19-moverscore/releases/download/0.6/MNLI_BERT.zip
unzip MNLI_BERT.zip -d bert-mnli
rm MNLI_BERT.zip
```

Reproduce the results for WMT15-16

```
python reproduce_1516.py  --metric baryscore --model bert-base-uncased --dataset 15
python reproduce_1516.py  --metric baryscore --model nli --dataset 15
python reproduce_1516.py   --metric bertscore --model bert-base-uncased --dataset 15
python reproduce_1516.py  --metric moverscore --model nli --dataset 15
python reproduce_1516.py  --metric moverscore --model bert-base-uncased --dataset 15

python reproduce_1516.py  --metric baryscore --model bert-base-uncased --dataset 16
python reproduce_1516.py  --metric baryscore --model nli --dataset 16
python reproduce_1516.py   --metric bertscore --model bert-base-uncased --dataset 16
python reproduce_1516.py  --metric moverscore --model nli --dataset 16
python reproduce_1516.py  --metric moverscore --model bert-base-uncased --dataset 16
```


## Regarding metric implementation
We did some small change based on the original metric implementation to easier run them in our experiments.
E.g., added batch computation for BaryScore (line 82 in bary_score_re.py), model choice for MoverScore (line 52 in moverscore_re.py), etc. Those changes won't affect the metric performance.

The sources of the original implementation are:

**bert_score**: https://github.com/Tiiiger/bert_score/tree/master/bert_score

**moverscore_re.py**: https://github.com/AIPHES/emnlp19-moverscore/blob/master/moverscore.py

**bary_score_re.py**: https://github.com/PierreColombo/nlg_eval_via_simi_measures/blob/main/bary_score.py

## Regarding evaluation scripts
Similar to the metric implementation, we slightly modified the code based on the original ones:

**reproduce_18.py** in WMT18_BERT: 

https://github.com/Tiiiger/bert_score/blob/master/reproduce/get_wmt18_seg_results.py

**reproduce_17.py and mt_utils.py** in WMT17_Mover: 

https://github.com/AIPHES/emnlp19-moverscore/blob/master/examples/run_MT.py 

https://github.com/AIPHES/emnlp19-moverscore/blob/master/examples/mt_utils.py

**reproduce_1516.py** in WMT15-16_Bary:

https://github.com/PierreColombo/nlg_eval_via_simi_measures/blob/main/raw_score/score_analysis.ipynb

## Acknowlegements
This repo is based on the code and data from:

[MoverScore: Text Generation Evaluating with Contextualized
Embeddings and Earth Mover Distance](https://github.com/AIPHES/emnlp19-moverscore)

[BERTScore: Evaluating Text Generation with BERT](https://github.com/Tiiiger/bert_score)

[NLG evaluation via Statistical Measures of Similarity: BaryScore, DepthScore, InfoLM](https://github.com/PierreColombo/nlg_eval_via_simi_measures)


