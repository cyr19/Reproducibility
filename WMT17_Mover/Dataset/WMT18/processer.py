import pandas as pd

df = pd.read_csv('Dataset/WMT18_BERT/AllRefDA.csv', encoding='utf-16',header=0)

print(df.columns)

#print(df.groupby(['SystemID', 'SegmentID', 'TypeID', 'SourceLanguage',
       #'TargetLanguage']).groups)
df = df.drop(['UserID','TypeID','StartTime','EndTime'],axis=1)

numbers = []
for sets, df_grouped in df.groupby(['SystemID', 'SegmentID', 'SourceLanguage',
       'TargetLanguage']):
    #print(sets)
    numbers.append(len(df_grouped))

print(set(numbers))

