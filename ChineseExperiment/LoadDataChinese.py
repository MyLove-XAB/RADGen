import pandas as pd
import time
import tensorflow as tf
import re
from sklearn.utils import shuffle


data = pd.read_csv("/share/home/xuaobo/ai_studio/tasks/data/ChineseData/preprocessed_data.csv")
print("数据总量：", len(data))

t1 = time.time()
text = [str(t) for t in data["article"].to_list()]
filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
oov_token = '<unk>'
doc_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=oov_token)
# print(len(text))
doc_tokenizer.fit_on_texts(text)
news = doc_tokenizer.texts_to_sequences(text)
# title也应该作为筛选条件

t2 = time.time()
print("编码时间：", (t2-t1)/60)
news_df = pd.Series(len(x) for x in news)
print("新闻全部数据信息：")
print(news_df.describe())  # 描述性统计
"""
    新闻全部信息： 
        count    201227.000000
        mean        704.886342
        std         772.112828
        min           1.000000
        25%         121.000000
        50%         480.000000
        75%        1018.000000
        max       16618.000000
"""
print("--------------------------------------")

doc = []
for i in text:
    a = re.split(r'。|！|？|!|\?|……', i)
    # a.pop() # pop the blank in tail
    doc.append(a)
res = []

sen_num_df = pd.Series([len(x) for x in doc])
print("句子个数的全部信息：")
print(sen_num_df.describe())
"""
    句子个数的全部信息：
        count    201227.000000
        mean         27.779383
        std          30.945458
        min           1.000000
        25%           6.000000
        50%          18.000000
        75%          39.000000
        max         749.000000
"""


for a in doc:  # 每一篇文章
    for x in a:  # 每一句话
        x_ = x.split(' ')
        res.append(len(x_)-1)
sen_df = pd.Series(res)
print("每个句子单词数的全部信息：")
print(sen_df.describe())
"""
    每个句子单词数的全部信息：
        count    5.589962e+06
        mean     2.583654e+01
        std      3.161342e+01
        min      0.000000e+00
        25%      1.500000e+01
        50%      2.300000e+01
        75%      3.300000e+01
        max      1.319800e+04
"""

print("--------------------------------------")


# 处理标题
title = data["title"]
filter = '#&/[\\]^_`~\t\n'
summary_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=filter, oov_token=oov_token)
summary_tokenizer.fit_on_texts(title)
headline = summary_tokenizer.texts_to_sequences(title)
headline_df = pd.Series(len(x) for x in headline)
print("标题全部信息：")
print(headline_df.describe())
"""
    标题全部信息：
        count    201227.000000
        mean         15.186411
        std           5.420422
        min           1.000000
        25%          12.000000
        50%          14.000000
        75%          18.000000
        max          56.000000
"""

# 筛选单词个数少于100的新闻
local_ls_1 = []
for i, x in enumerate(news):
    if (len(x) >= 20) and (len(x) <= 1500):
        local_ls_1.append(i)
local_ls_2 = []
for i, x in enumerate(headline):
    if (len(x) > 3) and (len(x) <= 40):
        local_ls_2.append(i)
print(len(local_ls_1))  # 177742  # 177749
print(len(local_ls_2))  # 201089  # 200680
local = set(local_ls_1) & set(local_ls_2)  # 177630  # 177323
print(len(local))

local_ls = list(local)
all_df_select = data.iloc[local_ls].reset_index()
all_df_select.to_csv("/share/home/xuaobo/ai_studio/tasks/data/ChineseData/filtered_data.csv")

text = [str(t) for t in all_df_select["article"].to_list()]
headline = [str(t) for t in all_df_select["title"].to_list()]

news = doc_tokenizer.texts_to_sequences(text)
headline = summary_tokenizer.texts_to_sequences(headline)

# title也应该作为筛选条件

t2 = time.time()
print("编码时间：", (t2-t1)/60)
news_df = pd.Series(len(x) for x in news)
headline_df = pd.Series(len(x) for x in headline)
print("筛选后新闻标题的信息：")
print(headline_df.describe())
"""
    筛选后新闻标题的信息：
        count    177323.000000
        mean         14.955809
        std           5.116190
        min           4.000000
        25%          12.000000
        50%          14.000000
        75%          18.000000
        max          40.000000
"""

print("筛选后新闻全部数据信息：")
print(news_df.describe())  # 描述性统计

"""
    筛选后新闻全部数据信息：
        count    177323.000000
        mean        494.405046
        std         416.877970
        min          20.000000
        25%         103.000000
        50%         369.000000
        75%         810.000000
        max        1500.000000
"""

doc = []
for i in text:
    a = re.split(r'。|！|？|!|\?|……', i)
    # a.pop()  # pop the blank in tail
    doc.append(a)

sen_num_df = pd.Series([len(x) for x in doc])
print("句子个数的全部信息：")
print(sen_num_df.describe())
"""
    句子个数的全部信息：
        count    177323.000000
        mean         19.698652
        std          16.446919
        min           1.000000
        25%           5.000000
        50%          14.000000
        75%          31.000000
        max         143.000000
"""

res = []
for a in doc:  # 每一篇文章
    for x in a:  # 每一句话
        x_ = x.split(' ')
        res.append(len(x_)-1)
sen_df = pd.Series(res)
print("平均每篇文章句子长度全部信息：")
print(sen_df.describe())

"""
    平均每篇文章句子长度全部信息：
        count    3.493024e+06
        mean     2.557427e+01
        std      2.149336e+01
        min      0.000000e+00
        25%      1.500000e+01
        50%      2.300000e+01
        75%      3.300000e+01
        max      2.192000e+03
"""

# all_df_new = all_df_select.sample(frac=1.0, random_state=111).reset_index() # 采样率为1，等价于shuffle. 采样0.6 相当于抽样+shuffle
all_df_new = shuffle(all_df_select, random_state=100).reset_index()


# 划分训练集和测试集
def split_data(data, train_frac, val_frac):
    train = data[:int(len(data)*train_frac)]
    val = data[int(len(data)*train_frac):int(len(data)*(train_frac+val_frac))]
    test = data[int(len(data)*(train_frac+val_frac)):]
    return train, val, test


# 考虑一下要不要先选10w条出来作为数据集，然后再划分训练集、验证集、测试集
train_select, val_select, test_select = split_data(all_df_new, train_frac=0.92, val_frac=0.04)
train_select = train_select.drop(columns=["level_0"]).reset_index()
val_select = val_select.drop(columns=["level_0"]).reset_index()
test_select = test_select.drop(columns=["level_0"]).reset_index()
train_select_ = train_select.drop(columns=["level_0", "index", "Unnamed: 0"])
val_select_ = val_select.drop(columns=["level_0", "index", "Unnamed: 0"])
test_select_ = test_select.drop(columns=["level_0", "index", "Unnamed: 0"])
print("训练集长度：", len(train_select))  # 159867  # 163137
print("验证集长度：", len(val_select))  # 8881  # 7093
print("测试集长度：", len(test_select))  # 8882  # 7093
train_select_.to_csv('/share/home/xuaobo/ai_studio/tasks/data/ChineseData/train_select.csv', encoding="utf-8")
val_select_.to_csv('/share/home/xuaobo/ai_studio/tasks/data/ChineseData/val_select.csv', encoding="utf-8")
test_select_.to_csv('/share/home/xuaobo/ai_studio/tasks/data/ChineseData/test_select.csv', encoding="utf-8")
