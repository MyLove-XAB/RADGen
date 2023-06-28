import jieba
import pandas as pd
from tqdm import tqdm
import re


HMM_ = False

with open("/share/home/xuaobo/ai_studio/tasks/cnndm_test/papercode_small_py/ChineseExperiment/stopword.txt") as f:
    stopwords = f.read().splitlines()
stopWord = set(stopwords)
with open("/share/home/xuaobo/ai_studio/tasks/cnndm_test/papercode_small_py/ChineseExperiment/specialword.txt", 'r',
          encoding='utf8') as f:
    lines = f.readlines()
    [jieba.add_word(line.split()[0], freq=9999) for line in lines]

all_data_select = pd.read_csv("/share/home/xuaobo/ai_studio/tasks/data/ChineseData/all_data_select.csv")

filter_word = set(list([" ", "\xa0", "\u200b", "#", "&", "/", "\t", "\n", "[", "\\", "]", "^", "_", "`", "~"]))
tokenized_res = []
for article in tqdm(all_data_select["article_raw"]):
    # print(article)
    tmp = []
    for word in jieba.cut(str(article), HMM=HMM_):  # HMM=True, 能把小数什么的分出来
        if word in stopWord:
            continue
        else:
            word = re.sub(r"[|｜│丨︱／●]", "。", word)
            word = re.sub(r"\.{3}|\.{6}", "……", word)
            tmp.append(word)
    # print(tmp)

    tokenized_res.append(" ".join(tmp))  # 空格分隔
    # print(tokenized_res[-1])
    # print("------------------------------------")

tokenized_tres = []
raw_title = []
for title in tqdm(all_data_select["clean_title"]):
    tmp = []
    for word in jieba.cut(title, HMM=HMM_):
        if word in filter_word:
            continue
        else:
            word = re.sub(r"[|｜│丨︱/／]", "，", word)
            word = re.sub(r"\.{3}|\.{6}", "……", word)
            tmp.append(word)
    # print(tmp)
    # print("-----------------------")
    tokenized_tres.append(tmp)

tokenized_titles = []
for title in tokenized_tres:
    tokenized_titles.append(str(" ".join(title)))
    raw_title.append("".join(title))
# print(len(tokenized_titles))

# print(filter_word)
data = pd.DataFrame()
# data["raw_title"] = all_data_select["title"]
data["article"] = tokenized_res
data["title"] = tokenized_titles
data["raw_article"] = all_data_select["article_raw"]
data["raw_title"] = raw_title
# data["raw_article"] = all_data_select["detail"]
data.to_csv("/share/home/xuaobo/ai_studio/tasks/data/ChineseData/preprocessed_data.csv")
