"""
    calculate some training data ROUGE metrics as baseline
    notice: tokenizer should be same as TrainingModel's tokenizer, therefore, must use same data to generate tokenizer

"""

import numpy as np
import pandas as pd
from rouge_new import rouge_score
import tensorflow as tf
import re
import time
from TransformerPy import Transformer, create_padding_mask, create_look_ahead_mask
from tqdm import tqdm

ep_num = 5
num_layers = 2
d_model = 128
dff = 256
num_heads = 4  # 下次训练的时候减少heads的数量
para_size = 512

print("load training data")
news = pd.read_csv('/share/home/xuaobo/ai_studio/tasks/data/ChineseData/train_select.csv')
news = news.drop(['Unnamed: 0'], axis=1)
TRAIN_SIZE=len(news)
DATA_SIZE = TRAIN_SIZE
document = [str(x) for x in news['article'][:TRAIN_SIZE]]
summary1 = news['title'][:TRAIN_SIZE]
summary = summary1.apply(lambda x: '<go> ' + x + ' <stop>')

# tokenizing the texts into integers
filters = '#&/[\\]｜^_`~\t\n'  # 将, < > 保留下来
oov_token = '<unk>'
doc_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=oov_token)
summary_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=filters, oov_token=oov_token)
doc_tokenizer.fit_on_texts(document)
summary_tokenizer.fit_on_texts(summary)

# set vocabulary size
encoder_vocab_size = len(doc_tokenizer.word_index) + 1
decoder_vocab_size = len(summary_tokenizer.word_index) + 1
print("encoder_vocab_size: ", encoder_vocab_size, "; decoder_vocab_size: ", decoder_vocab_size)  # 304066; 50336

doc = []
for i in document:  # use test data
    a = re.split(r'。|！|？|……|!|\?', i)
    a.pop()  # pop the blank in tail
    doc.append(a)

# set max encoder_sentence_length and target_length
encoder_maxlen = 32
decoder_maxlen = 20 # target length

# Padding/Truncating sequences to identical sequence lengths
# first pad each sentences
doc_inputs = []
# del_lis = []
for id, x in enumerate(doc):
    inputs = doc_tokenizer.texts_to_sequences(x)
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=encoder_maxlen, padding='post', truncating='post')
    doc_inputs.append(inputs)


# reshape sentence to (-1) 合并然后降成一维
doc_inputs_new = []
for doc in doc_inputs:
    a = np.reshape(doc, (-1))
    doc_inputs_new.append(a)

# padding again to 30 sentences, total embedding size: 16*40=640
inputs = tf.keras.preprocessing.sequence.pad_sequences(doc_inputs_new, maxlen=encoder_maxlen, padding='post', truncating='post')
inputs = tf.cast(inputs, dtype=tf.int32)
print("inputs[0].shape: ", inputs[0].shape)
print("finish padding inputs number: ", len(inputs))


# get transformer result
def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


def transSum(input_document, env, decoder_maxlen):
    """

        :param input_document: a series document
        :param env: transformer
        :param decoder_maxlen: max headline length
        :return:
        """
    # encoder_input = tf.expand_dims(input_document, 0)  # 对input_document[0]加一个第0维，这样encoder_input就是shape [1*120]
    encoder_input = input_document
    decoder_input = [summary_tokenizer.word_index["<go>"] for _ in range(len(input_document))]
    output = tf.expand_dims(decoder_input, 1)  # [para_size, 1]
    # output = decoder_input

    for i in range(decoder_maxlen):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        predictions, attention_weights = env(encoder_input, output, False,
                                             enc_padding_mask, combined_mask, dec_padding_mask)

        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        # if predicted_id == summary_tokenizer.word_index["<stop>"]:
        #     return tf.squeeze(output, axis=0), attention_weights  # , ac_sample # tf.squeeze 删除axis=0，且维度为1 的维度

        output = tf.concat([output, predicted_id], axis=-1)
    return output, attention_weights  # , ac_sample

transformer = Transformer(num_layers, d_model, num_heads, dff, encoder_vocab_size,
                          decoder_vocab_size, pe_input=encoder_vocab_size, pe_target=decoder_vocab_size,)

transformer.load_weights("/share/home/xuaobo/ai_studio/tasks/weights/weights_Chinese/weights1_epoch{}.ckpt".format(ep_num))
print('loaded transformer weights')

rouge_base = []
# size = len(inputs) # which is equal to DATA_SIZE
t1 = time.time()
for i in tqdm(range(len(inputs)//para_size + 1)):
    doc = inputs[i * para_size:min((i + 1) * para_size, len(inputs))]  # [para_size, 640]
    reference = news['title'][i * para_size:min((i + 1) * para_size, len(inputs))].to_list()
    res_trans, _ = transSum(doc, transformer, decoder_maxlen)

    # 转化成文本，计算reward
    for ref_ind, generate_sequence in enumerate(res_trans):
        for ind, sequence in enumerate(generate_sequence):
            if sequence == summary_tokenizer.word_index["<stop>"]:
                generate_sequence = generate_sequence[:ind]
                break
        trans_summary = summary_tokenizer.sequences_to_texts(np.expand_dims(generate_sequence.numpy()[1:], 0))[0]
        # [[1 26 221]]需要是这种形式才能够texts_to_sequences
        if len(trans_summary) == 0:  # 没生成文章
            rouge_base.append([0., 0., 0.])
            continue
        r1f, r2f, rlf = rouge_score(trans_summary, reference[ref_ind])  # 计算奖励
        rouge_base.append([r1f, r2f, rlf])

    print("iters: {}, takes seconds: {}".format(i, (time.time() - t1)))
    t1 = time.time()

base = pd.DataFrame(rouge_base, columns=['r1f', 'r2f', 'rlf'])
base.to_csv('/share/home/xuaobo/ai_studio/tasks/res/Chinese/train_baseline1_ep{}.csv'.format(ep_num))
print("Save Results")
