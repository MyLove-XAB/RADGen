# -*- coding = utf-8 -*-
# @Time: 2022/7/16 20:27
# @Author: Aobo
# @File: rouge_new.py
# @Software: PyCharm

"""
   Calculate ROUGE metrics
"""

from rouge import Rouge
import numpy as np
rouge = Rouge()


def rouge_score(pred, ref):
    rls = rouge.get_scores(hyps=pred, refs=ref)
    r = rls[0]
    r1f = r['rouge-1']['f'] # 'p', 'r'
    r2f = r['rouge-2']['f']
    rlf = r['rouge-l']['f']
    return r1f, r2f, rlf


if __name__ == "__main__":
    pred = "police kill , the gunman"
    ref = "the gunman kill police"
    print(rouge_score(pred, ref))

    pred = "police kill gunman"
    ref = "police kill gunmans"
    print(rouge_score(pred, ref))

    pred = "我 爱 北京"
    ref = "我 爱 上海"
    print(rouge_score(pred, ref))

    for itr in range(5):  # batch 更新，每遍历一次n_iter则遍历一遍全部训练数据
        print(np.random.randint(0, 1000 - 10))
    print(np.linspace(1, 40, num=40))
""" 
    经测试 标点符号回影响计算结果 而且标点符号和单词之间有无空格也会影响
    中文的通过jieba分词 应该会自动把标点符号分开，但是英文则需要看一下数据集了
"""