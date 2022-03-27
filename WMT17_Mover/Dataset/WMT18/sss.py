import pandas as pd

data = pd.read_csv('RR-seglevel.csv',header=0,delimiter=' ')
print(data.columns)
print(data)
lps = set(data.index)
print(lps)

for lp in lps:
    tmp = data.loc[[lp],['SID']]
    print("{}: {}".format(lp, len(set(tmp['SID']))))
