# Reproducibility Issues for BERT-based Evaluation Metrics

##Reproduction on MT
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

## Clarification
###Regarding metric implementation
We did some small change based on the original metric implementation to better run them in our experiments.
E.g., added batch computation for BaryScore (line 84 in bary_score_re.py), model choice for MoverScore (line 52 in moverscore_re.py), etc. Those changes won't affect the metric performance.

The sources of the original implementation are:

bert_score: https://github.com/Tiiiger/bert_score/tree/master/bert_score

moverscore_re.py: https://github.com/AIPHES/emnlp19-moverscore/blob/master/moverscore.py

bary_score_re.py: https://github.com/PierreColombo/nlg_eval_via_simi_measures/blob/main/bary_score.py

###Regarding evaluation scripts:
Similar to the metric implementation, we slightly modified the code based on the original:

reproduce_18.py in WMT18_BERT: 

https://github.com/Tiiiger/bert_score/blob/master/reproduce/get_wmt18_seg_results.py

reproduce_17.py, mt_utils.py and wmt_eval.py in WMT17_Mover: 

https://github.com/AIPHES/emnlp19-moverscore/blob/master/examples/run_MT.py 

https://github.com/AIPHES/emnlp19-moverscore/blob/master/examples/mt_utils.py

reproduce_1516.py in WMT15-16_Bary:

https://github.com/PierreColombo/nlg_eval_via_simi_measures/blob/main/raw_score/score_analysis.ipynb


