##
import re
from collections import Counter

import jieba
import pandas as pd

import sys

##
df = pd.read_csv('../data/data.csv')

l = []
jb = []
maxlen = 200
for index, row in df.iterrows():
    if pd.isna(row['影像表现']):
        pass
    elif pd.isna(row['检查诊断']):
        pass
    elif row['影像表现'] == '' or row['检查诊断'] == '':
        pass
    else:
        f = row['影像表现'].replace('\n', ' ').replace('　', ' ').replace('\t', ' ').replace('@', '').replace(',', '，')
        f = re.sub(' +', '', f)
        if 1 < len(f) <= 200:
            l.append(f)
            seg_list_f = jieba.cut(f, cut_all = False)
            jb.append(str("/ ".join(seg_list_f)))
        d = row['检查诊断'].replace('\n', ' ').replace('　', ' ').replace('\t', ' ').replace('@', '').replace(',', '，')
        d = re.sub(' +', '', d)
        if 1 < len(d) <= 200:
            l.append(d)
            seg_list_d = jieba.cut(d, cut_all = False)
            jb.append(str("/ ".join(seg_list_d)))
    if index >= 5000:
        break

length = []
for i in l:
    # print(i)
    length.append(len(i))

for i in jb:
    print(i)

print(max(length))
freq = dict(Counter(length))
for index, data_dict in enumerate(sorted(freq.items(), key = lambda d: d[0], reverse = True)):
    print(str(index + 1) + ':', data_dict, '\t', end = "")
    if (index + 1) % 10 == 0:
        print()
print()
# for index, data_dict in enumerate(sorted(freq.items(), key = lambda d: d[1], reverse = True)):
#     print(str(index + 1) + ':', data_dict, '\t', end = "")
#     if (index + 1) % 10 == 0:
#         print()
# print()
dct = {
    '分词': jb,
    '原文': l,
}
pd_export = pd.DataFrame(dct)
# print(pd_export)
pd_export.to_csv('../data/data_export.csv', encoding = 'utf-8')

