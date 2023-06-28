# -*- coding: UTF-8 -*-
"""
    Training Model for Headline Transformer
    package version:    tf 2.4.1
    data:   Chinese all data 30句话每句32个词
    hyper parameters:   encoder_maxlen = 32
                        decoder_maxlen = 20
                        BUFFER_SIZE = 300000
                        BATCH_SIZE = 128
                        # hyper params
                        num_layers = 2
                        d_model = 128
                        dff = 256
                        num_heads = 4  # 下次训练的时候减少heads的数量
                        EPOCHS = 20
                        warm_start_steps = 5000
    ** 每次运行注意修改路径中 version 版本 **
    保留标题中的部分标点符号
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import time
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from TransformerPy import Transformer, create_padding_mask, create_look_ahead_mask, create_masks, CustomSchedule
import sys

sys.path.append("../")

print("load training data")
print(tf.test.is_gpu_available())
news = pd.read_csv('/share/home/xuaobo/ai_studio/tasks/data/ChineseData/train_select.csv')
TRAIN_SIZE = len(news)  # -1: all data - 1
news = news.drop(['Unnamed: 0'], axis=1)
document = [str(x) for x in news['article'][:TRAIN_SIZE]]
summary1 = news['title'][:TRAIN_SIZE]
summary = summary1.apply(lambda x: '<go> ' + x + ' <stop>')

doc = []
for i in document:
    a = re.split(r'。|!|\?|！|？|……', i)
    a.pop()  # pop the blank in tail
    doc.append(a)

# tokenizing the texts into integers
filters = '#&/[\\]^_`｜~\t\n'  # 保留, ! ?，对标题进行断句，如果没有这句，会把下面<go>中的尖括号过滤掉
oov_token = '<unk>'
doc_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=oov_token)
summary_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=filters, oov_token=oov_token)
doc_tokenizer.fit_on_texts(document)
summary_tokenizer.fit_on_texts(summary)
targets0 = summary_tokenizer.texts_to_sequences(summary)

# set vocabulary size
encoder_vocab_size = len(doc_tokenizer.word_index) + 1
decoder_vocab_size = len(summary_tokenizer.word_index) + 1
print("encoder_vocab_size: ", encoder_vocab_size, "; decoder_vocab_size: ", decoder_vocab_size)  # (252622, 46263)

"""
    some statistical result:
        document length: mean: 421, min: 7, 50%: 380, 75%: 530, max: 1897
        target length: max: 53, 75%: 27， mean: 22.997278
        sentence number: mean: 36.361243, min:6, max: 360, 50%: 32, 75%: 45, max: 352
        sentence lengths: mean: 11.92896, 50%: 12, 75%: 16, max: 788
"""

# set max encoder_sentence_length and target_length
encoder_maxlen = 32  # sentence length
decoder_maxlen = 20  # target length
# sen_num = 40

# Padding/Truncating sequences to identical sequence lengths
# first pad each sentences


def text_padding(doc, doc_tokenizer, targets):
    doc_inputs, del_lis = [], []

    for id, x in enumerate(doc):
        x_ = doc_tokenizer.texts_to_sequences(x)
        x_ = tf.keras.preprocessing.sequence.pad_sequences(x_, maxlen=encoder_maxlen, padding='post',
                                                           truncating='post')
        doc_inputs.append(x_)
    # pad targets
    targets = tf.keras.preprocessing.sequence.pad_sequences(targets, maxlen=decoder_maxlen, padding='post',
                                                            truncating='post')
    # reshape sentence to (-1) 合并然后降成一维
    doc_inputs_new = []
    for doc in doc_inputs:
        a = np.reshape(doc, (-1))
        doc_inputs_new.append(a)

    # padding again to 30 sentences, total embedding size: 40*16= 640
    inputs = tf.keras.preprocessing.sequence.pad_sequences(doc_inputs_new, maxlen=32 * 30, padding='post',
                                                           truncating='post')
    inputs = tf.cast(inputs, dtype=tf.int32)
    targets = tf.cast(targets, dtype=tf.int32)
    print("finish padding inputs number: ", len(inputs))

    return inputs, targets


inputs, targets = text_padding(doc=doc, doc_tokenizer=doc_tokenizer, targets=targets0)


BUFFER_SIZE = 300000  # 对于完美的洗牌，需要大于或等于数据集的完整大小的缓冲区大小
BATCH_SIZE = 128
dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print("finish construct train dataset, BATCH_SIZE: ", BATCH_SIZE)


# load val data
val_news = pd.read_csv('/share/home/xuaobo/ai_studio/tasks/data/ChineseData/val_select.csv')
val_news = val_news.drop(['Unnamed: 0'], axis=1)
VAL_SIZE = len(val_news)
val_document = [str(x) for x in val_news['article'][:VAL_SIZE]]
val_summary1 = val_news['title'][:VAL_SIZE]
val_summary = val_summary1.apply(lambda x: '<go> ' + x + ' <stop>')

val_doc = []
for i in val_document:
    a = re.split(r'。|！|？|……|!|\?', i)
    a.pop()  # pop the blank in tail
    val_doc.append(a)

# tokenizing the texts into integers
val_targets0 = summary_tokenizer.texts_to_sequences(val_summary)

val_inputs, val_targets = text_padding(doc=val_doc, doc_tokenizer=doc_tokenizer, targets=val_targets0)
val_dataset = tf.data.Dataset.from_tensor_slices((val_inputs, val_targets)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# hyper params
num_layers = 2
d_model = 128
dff = 256
num_heads = 4  # 下次训练的时候减少heads的数量
EPOCHS = 30


# Defining losses and other metrics
learning_rate = CustomSchedule(d_model)  # instantiate
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
# loss_object
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


step_loss = tf.keras.metrics.Mean(name='train_loss')
# Instantiate
transformer = Transformer(num_layers, d_model, num_heads, dff, encoder_vocab_size,
                          decoder_vocab_size, pe_input=encoder_vocab_size, pe_target=decoder_vocab_size, )
# transformer.summary()


# directory path
checkpoint_path = "/share/home/xuaobo/ai_studio/tasks/weights/weights_Chinese/checkpoints_Chinese"
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
#  load previous checkpoint result
# if ckpt_manager.latest_checkpoint:
#     ckpt.restore(ckpt_manager.latest_checkpoint)
#     print('Latest checkpoint restored!!')


# training steps
@tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, True,
                                     enc_padding_mask, combined_mask, dec_padding_mask)  # call
        loss = loss_function(tar_real, predictions)  # 输入多少个tar_inp，输出多少个predictions

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    step_loss(loss)  # 可以记录之前的loss的均值，通过reset_states()可以重置清空


val_loss = tf.keras.metrics.Mean(name='val_loss')


def val_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    predictions, _ = transformer(inp, tar_inp, False, enc_padding_mask, combined_mask, dec_padding_mask)  # call
    loss = loss_function(tar_real, predictions)
    val_loss(loss)


epoch_list = []
val_list = []
# epoch_loss = tf.keras.metrics.Mean(name='epoch_loss')
for epoch in tqdm(range(EPOCHS)):
    start = time.time()
    step_loss.reset_states()
    val_loss.reset_states()

    for (batch, (inp, tar)) in enumerate(dataset):
        train_step(inp, tar)

        if batch % 500 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, step_loss.result()))

    # epoch_loss(step_loss.result()) # 用不着计算所有epoch得到的loss的均值
    epoch_list.append(step_loss.result())

    # if (epoch + 1) % 2 == 0:
    ckpt_save_path = ckpt_manager.save()
    print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
    transformer.save_weights('/share/home/xuaobo/ai_studio/tasks/weights/weights_Chinese/weights1_epoch{}.ckpt'.format(epoch+1))

    # val_dataset validation
    for (batch, (inp, tar)) in enumerate(val_dataset):
        val_step(inp, tar)
    val_list.append(val_loss.result())

    print('Epoch {} Loss {:.4f}'.format(epoch + 1, step_loss.result()))
    print('Val Epoch {} Loss {:.4f}'.format(epoch + 1, val_loss.result()))
    print('Time taken for 1 epoch: {} min\n'.format((time.time() - start)/60))

print(transformer.summary())


# save and plot training loss
df = pd.DataFrame(range(len(epoch_list)), columns=['epoch'])
df['epoch_loss'] = pd.Series(epoch_list)
df.to_csv('/share/home/xuaobo/ai_studio/tasks/res/Chinese/training_loss.csv')

val_df = pd.DataFrame(range(len(val_list)), columns=['epoch'])
val_df['epoch_loss'] = pd.Series(val_list)
df.to_csv('/share/home/xuaobo/ai_studio/tasks/res/Chinese/validation_loss.csv')

plt.plot(df['epoch'], df['epoch_loss'], label="training loss")
plt.plot(val_df['epoch'], val_df['epoch_loss'], label="val loss")
plt.legend()
plt.savefig(fname="/share/home/xuaobo/ai_studio/tasks/res/Chinese/loss.png", figsize=[10, 10])
plt.show()
