# Reproducibility

## Reproduction on WMT18

```
cd WMT18_BERT
sh download_wmt18.sh

python reproduce_18.py  --metric baryscore
python reproduce_18.py  --metric moverscore
python reproduce_18.py   --metric bertscore
```
## Reproduction on WMT17
```
cd WMT17_Mover

python reproduce_17.py  --metric baryscore --model bert-base-uncased
python reproduce_17.py   --metric bertscore --model bert-base-uncased
python reproduce_17.py  --metric moverscore --model bert-base-uncased
python reproduce_17.py  --metric moverscore --model original 
```

## Reproduction on WMT15-16
```
cd WMT15-16_Bary

python reproduce_1516.py  --metric bertscore --model bert-base-uncased --dataset 15
python reproduce_1516.py  --metric moverscore --model bert-base-uncased --dataset 15
python reproduce_1516.py  --metric baryscore --model bert-base-uncased --dataset 15
python reproduce_1516.py  --metric bertscore --model original --dataset 15
python reproduce_1516.py  --metric moverscore --model original --dataset 15

python reproduce_1516.py  --metric bertscore --model bert-base-uncased --dataset 16
python reproduce_1516.py  --metric moverscore --model bert-base-uncased --dataset 16
python reproduce_1516.py  --metric baryscore --model bert-base-uncased --dataset 16
python reproduce_1516.py  --metric bertscore --model original --dataset 16
python reproduce_1516.py  --metric moverscore --model original --dataset 16
