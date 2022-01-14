from random import shuffle

import pandas as pd

df = pd.read_csv("../data/labeled.csv")

l = []
for index, row in df.iterrows():
    if row["标签"] == 4:
        l.append(row["原文"].strip() + '\t' + '0')
    else:
        l.append(row["原文"].strip() + '\t' + str(row["标签"]))

shuffle(l)
with open('../data/train.txt', 'a', encoding = 'utf-8') as f1:
    with open('../data/val.txt', 'a', encoding = 'utf-8') as f2:
        with open('../data/test.txt', 'a', encoding = 'utf-8') as f3:
            for i, item in enumerate(l):
                if i <= len(l) * 0.8:
                    f1.write(item + '\n')
                elif (i > len(l) * 0.8) and (i <= len(l) * 0.9):
                    f2.write(item + '\n')
                else:
                    f3.write(item + '\n')
