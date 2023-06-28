"""
    to make it easy, we do not use RLAgent.py, because transfer it from ipynb need more time to modify.
    with dropout and layer normalizaion in RLAgent
    reward = r1f+r2f+r3f
"""
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import time
import re
import random
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange
import logging as log

from TransformerPy import Transformer, create_masks
# from RLAgent import RLActor
from rouge_new import rouge_score
import wandb

os.environ["WANDB_API_KEY"] = 'key'  # 将引号内的+替换成自己在wandb上的一串值
os.environ["WANDB_MODE"] = "offline"   # 离线  （此行代码不用修改）
wandb.init(project="ChineseRL")

print("load training data")
news = pd.read_csv('/share/home/xuaobo/ai_studio/tasks/data/ChineseData/train_select.csv')
val_news = pd.read_csv('/share/home/xuaobo/ai_studio/tasks/data/ChineseData/val_select.csv')
# hyper parameters
TRAIN_SIZE = len(news)
RL_TRAIN_SIZE = len(news)
VAL_SIZE = len(val_news)  # len(val_news) # len(val_news)
VAL_POS = 0
base = 0.
times = 10

ep_num = 5
num_layers = 2
d_model = 128
dff = 256
num_heads = 4  # 下次训练的时候减少heads的数量

sen_dim = 32
output_dim = 512  # RL模型中的神经网络层神经元个数
sen_num = 30

batch_size_rl = 256  # 涉及到RLmodule的更新, 之前是512
para_size = 256  # 并行计算reward
learning_rate = 0.001
n_experiment = 100  # 180 # 只跑一次 多次的话 每次需要加载rlagent的参数
# n_iter = RL_TRAIN_SIZE // batch_size_rl

# wandb.init(project="RLTG_new1")
# config.n_iter = 100000//config.batch_size_rl

#  get tokenizer
print(tf.test.is_gpu_available())
news = news.drop(['Unnamed: 0'], axis=1)
news = news[:TRAIN_SIZE]

train_reward_df = pd.read_csv('/share/home/xuaobo/ai_studio/tasks/res/Chinese/train_baseline1_ep{}.csv'.format(ep_num))  # used for val
tr_r1 = train_reward_df['r1f']
tr_r2 = train_reward_df['r2f']
tr_r3 = train_reward_df['rlf']
train_base_reward = tr_r1 + tr_r2 + tr_r3
train_base_reward1 = train_base_reward[:RL_TRAIN_SIZE]

document = news['article']
summary1 = news['title']
summary = summary1.apply(lambda x: '<go> ' + x + ' <stop>')

reward_df = pd.read_csv('/share/home/xuaobo/ai_studio/tasks/res/Chinese/val_baseline1_ep{}.csv'.format(ep_num))  # used for val
base_r1 = reward_df['r1f']
base_r2 = reward_df['r2f']
base_r3 = reward_df['rlf']
base_reward = base_r1 + base_r2 + base_r3
base_reward1 = base_reward[VAL_POS:VAL_POS+VAL_SIZE].reset_index()

doc = []
for i in document:
    a = re.split(r'。|！|？|……|!|\?', i)
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
# doc_tokenizer.word_index

# set vocabulary size
encoder_vocab_size = len(doc_tokenizer.word_index) + 1
decoder_vocab_size = len(summary_tokenizer.word_index) + 1
print("encoder_vocab_size: ", encoder_vocab_size, "; decoder_vocab_size: ", decoder_vocab_size)


# set max encoder_sentence_length and target_length
encoder_maxlen = 32  # sentence length
decoder_maxlen = 20  # target length
# sen_num = 40

# Padding/Truncating sequences to identical sequence lengths
# first pad each sentences
doc_inputs = []
del_lis = []
for id, x in enumerate(doc):
    inputs = doc_tokenizer.texts_to_sequences(x)
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=encoder_maxlen, padding='post', truncating='post')
    doc_inputs.append(inputs)
# pad targets
targets = tf.keras.preprocessing.sequence.pad_sequences(targets0, maxlen=decoder_maxlen, padding='post', truncating='post')

# reshape sentence to (-1) 合并然后降成一维
doc_inputs_new = []
for doc in doc_inputs: # 注意这里用了doc，所以for语句执行完之后，后面的doc就值的是doc_inputs的最后一个
    a = np.reshape(doc, (-1))
    doc_inputs_new.append(a)

# padding again to 30 sentences, total embedding size: 30*28=840
inputs = tf.keras.preprocessing.sequence.pad_sequences(doc_inputs_new, maxlen=32 * 30, padding='post', truncating='post')
inputs = tf.cast(inputs, dtype=tf.int32)
targets = tf.cast(targets, dtype=tf.int32)
print("finish padding inputs number: ", len(inputs))


# load val data
val_news = pd.read_csv('/share/home/xuaobo/ai_studio/tasks/data/ChineseData/val_select.csv')
val_news = val_news.drop(['Unnamed: 0'], axis=1)
val_document = val_news['article'][VAL_POS:VAL_POS+VAL_SIZE]
val_summary = val_news['title'][VAL_POS:VAL_POS+VAL_SIZE]
# val_summary = val_summary1.apply(lambda x: '<go> ' + x + ' <stop>')

# split and padding
val_doc = []
for i in val_document:
    a = re.split(r'。|！|？|……|!|\?', i)
    a.pop()  # pop the blank in tail
    val_doc.append(a)

val_doc_inputs = []
# val_del_lis = []
for id, x in enumerate(val_doc):
    val_inputs = doc_tokenizer.texts_to_sequences(x)
    val_inputs = tf.keras.preprocessing.sequence.pad_sequences(val_inputs, maxlen=encoder_maxlen, padding='post',
                                                               truncating='post')
    val_doc_inputs.append(val_inputs)

# reshape sentence to (-1) 合并然后降成一维
val_doc_inputs_new = []
for doc in val_doc_inputs:
    a = np.reshape(doc, (-1))
    val_doc_inputs_new.append(a)

# padding again to 30 sentences, total embedding size: 960
val_inputs = tf.keras.preprocessing.sequence.pad_sequences(val_doc_inputs_new,
                                                           maxlen=960, padding='post', truncating='post')
val_inputs = tf.cast(val_inputs, dtype=tf.int32)


def sample_ac(ac, test=False):
    ac = tf.math.sigmoid(ac)
    if test:
        ac_sample = tf.where(ac >= 0.5, 1, 0)  # 大于0.5为1，greedy
    else:
        # 注意看一下 RLActor 最后的输出有没有sigmoid激活
        nums = len(ac)
        out = tf.reshape(ac, [-1])
        # pseudorandom values from a binomial distribution.
        ac = np.random.binomial(n=1, p=out, size=len(out))  # random values
        ac = np.reshape(ac, [nums, -1])  # 并行需要
        ac_sample = tf.cast(ac, tf.int32)
    return ac_sample


def apply_ac(ac_sample, sen_dim, sen_num, doc):
    # notice that when evaluate BATCH_SIZE=1

    ac = tf.broadcast_to(ac_sample, [sen_dim, len(ac_sample), sen_num])
    ac = tf.transpose(ac, perm=[1, 2, 0])
    ac = tf.reshape(ac, [len(ac_sample), -1])
    inp = tf.multiply(doc, ac)
    return inp  # , ac_sample 方便记录ac


# 可以考虑定义成一个类 方便后面训练调用
def get_result(ac_sample, sen_dim, sen_num, input_document, env, decoder_maxlen=decoder_maxlen):
    input_document = apply_ac(ac_sample, sen_dim, sen_num, input_document)
    encoder_input = input_document  # tf.reshape(input_document, [-1, sen_num*sen_dim])
    decoder_input = [summary_tokenizer.word_index["<go>"] for i in range(len(input_document))]
    output = tf.expand_dims(decoder_input, 1)

    for i in range(decoder_maxlen):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        predictions, attention_weights = env(encoder_input, output, False,
                                             enc_padding_mask, combined_mask, dec_padding_mask)

        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        output = tf.concat([output, predicted_id], axis=-1)

    return output, attention_weights  # , ac_sample


def reward_function(reference, summary):
    """
        Calculate the reward between the reference and summary,
        which can be composed by different ROUGE measure
    """
    r1f, r2f, rlf = rouge_score(summary, reference)
    reward = r1f + r2f + rlf # + r3  # r1, r2, r3 的权重有待确定

    return reward


class RLActor(tf.keras.Model):
    """get action based on input document with simple NN"""

    def __init__(self, sen_num, output_dim, input_vocab_size, env, drate=0.2):
        # def __init__(self, sen_num, d_model, output_dim, input_vocab_size, drate=0.1):

        super(RLActor, self).__init__()
        self.sen_num = sen_num
        self.output_dim = output_dim
        #         self.action_dim = action_num
        #         self.sen_dim = sen_dim
        self.embed = tf.keras.layers.Embedding(input_vocab_size, d_model) # 试着换成transformer的embedding层
        self.batchnorm = tf.keras.layers.LayerNormalization()

        # self.embed = env.encoder.embedding # 用相同的embedding层
        # self.embed.trainable = False

        # 可以在这里加入MHA
        self.layer1 = layers.LSTM(self.output_dim, activation='tanh', return_state=True)  # tanh for LSTM
        self.layer2 = layers.Dense(self.output_dim, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(drate)
        self.layer3 = layers.Dense(self.output_dim/2, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(drate)
        self.layer4 = layers.Dense(self.sen_num)  # Multi-label: sigmoid     # , activation='sigmoid'

    def call(self, x, training=False):  # get an action for input document,training
        x = self.embed(x)
        x, _, _ = self.layer1(x)

        x = self.batchnorm(x) # layernorm
        x = self.layer2(x) + x
        x = self.dropout1(x, training=training)
        x = self.layer3(x)
        x = self.dropout2(x, training=training)
        out = self.layer4(x)
        return out


class Agent(tf.keras.Model):
    def __init__(self, env, batch_size_rl, learning_rate,  # seed,
                 sen_num, sen_dim, output_dim, input_vocab_size, drate=0.1): # inputs, summary
        super(Agent, self).__init__()
        self.env = env
        #         self.seed = seed
        self.sen_dim = sen_dim
        self.batch_size = batch_size_rl  # 设定 batch 大小
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.sen_num = sen_num
        self.model = RLActor(sen_num, output_dim, input_vocab_size, env, drate=drate)

    def policy_param(self, obs, training):
        return self.model(obs, training)  # obs is equal to sentences

    def sample_trajectories(self, pos, training=False):  # 直接抽取batch_size个 # , env
        sample_doc = inputs[pos: pos + self.batch_size] # [:, tf.newaxis, ...]
        sample_ref = summary1[pos: pos + self.batch_size].to_list()
        paths = []

        # 这里需要调整成并行计算
        acs_prob = self.policy_param(sample_doc, training=training)
        acs = sample_ac(acs_prob)
        preds, atten_weights = get_result(acs, self.sen_dim, self.sen_num,
                                          sample_doc, self.env, decoder_maxlen=decoder_maxlen)

        for ref_ind, generate_sequence in enumerate(preds):
            for ind, sequence in enumerate(generate_sequence):
                if sequence == summary_tokenizer.word_index["<stop>"]:
                    generate_sequence = generate_sequence[:ind]
                    break
            rl_summary = summary_tokenizer.sequences_to_texts(np.expand_dims(generate_sequence.numpy()[1:], 0))[0]
            if len(rl_summary) == 0:  # 没生成文章
                reward = 0
            else:
                reward = (reward_function(sample_ref[ref_ind], rl_summary) - train_base_reward1[pos + ref_ind] - base) * times
            path = {'ob': tf.reshape(sample_doc[ref_ind], [1, -1]), 'ac': acs[ref_ind], 'reward': reward}
            paths.append(path)

        return paths

    def log_prob(self, policy_param, acs):
        logits = policy_param
        # Computes the cross-entropy loss between true labels and predicted labels.
        acs = tf.reshape(acs, [self.batch_size, -1])
        log_prob = tf.keras.losses.binary_crossentropy(y_true=acs, y_pred=logits, from_logits=True) # need to test
        return log_prob

    @tf.function  # train = tf.function(train)
    def train_step(self, obs, acs, rew):
        # remember "@tf.function" tensorizes elements so it is better to do preprocessing calculations beforehead for convenience.
        # Here, it means caculate qs beforehead

        # @tf.function  # train = tf.function(train)
        # tf.function: Compiles a function into a callable TensorFlow graph. (deprecated arguments)
        def train(obs, acs, rew):
            with tf.GradientTape() as tape:
                policy_param = self.policy_param(obs, training=True) # 直接用前面计算得到的ac_prob呢
                log_prob = self.log_prob(policy_param, acs)
                objective = tf.reduce_mean(log_prob * rew)  # reward都是0 需要调整，(0.8 * rew + 0.2)
                # 注意这里的 log_prob 是如何影响梯度传播的

            training_vars = self.model.trainable_variables
            # env_training_vars = self.env.trainables
            grads = tape.gradient(objective, training_vars)
            self.optimizer.apply_gradients(zip(grads, training_vars))

        # tensorize variables so that there is no error under @tf.function
        obs = tf.cast(obs, dtype=tf.int32)
        acs = tf.cast(acs, dtype=tf.int32)
        rew = tf.cast(rew, dtype=tf.float32)
        train(obs, acs, rew)

    def save_model(self, num):
        path = '/share/home/xuaobo/ai_studio/tasks/weights/weights_Chinese/4_weights_rl_ep{}_{}.ckpt'.format(ep_num, num)
        self.model.save_weights(path)
        # print("save model in {}".format(path))


# train step
input_vocab_size = encoder_vocab_size
env1 = Transformer(num_layers, d_model, num_heads, dff, encoder_vocab_size,
                   decoder_vocab_size, pe_input=encoder_vocab_size, pe_target=decoder_vocab_size,)
# load weights to initiate env
env1.load_weights('/share/home/xuaobo/ai_studio/tasks/weights/weights_Chinese/weights1_epoch{}.ckpt'.format(ep_num))
agent = Agent(env1, batch_size_rl, learning_rate,  # seed,
              sen_num, sen_dim, output_dim, input_vocab_size, drate=0.1)


def train_PG(n_iter, batch_size_rl, n_exp):
    """
        try to let the agent independent from environment so that we can use the agent in other envs.
        And also think it is right concept because we could introduce other agents with the same env.
    """
    meanlis = []

    stop_index = summary_tokenizer.word_index["<stop>"]
    for itr in range(n_iter):  # batch 更新，每遍历一次n_iter则遍历一遍全部训练数据
        # t = time.time()
        pos = np.random.randint(0, RL_TRAIN_SIZE - batch_size_rl)  # 此时随机选择数据进行更新
        # if itr*(batch_size_rl+1) >= TRAIN_SIZE:
        #     break
        # pos = itr*batch_size_rl  # 此时遍历全部数据进行更新

        paths = agent.sample_trajectories(pos, training=True)
        obs = np.concatenate([path['ob'] for path in paths])
        acs = np.concatenate([path['ac'] for path in paths])
        rew = [path['reward'] for path in paths]
        agent.train_step(obs, acs, rew)

        # rewards = [res['reward'].sum() for res in paths] # 应该没有影响，因为只有一个reward
        os.system('clear')  # win: os.system('cls'); linux: os.system('clear')
        # train_reward(rew)
        meanRew = np.mean(rew)
        meanlis.append(meanRew)
        # log.info(time.time()-t)
    wandb.log({'mean training reward': np.mean(meanlis)/times + base})
    agent.save_model(n_exp + 1)
    # print('save experiment exp: {}'.format(n_exp + 1))

    # cal val_reward
    rouge_res_rl = []  # rl reward
    ac_res = []  # 记录rl模型的动作,后面根据训练的epoch 分别记录
    generate_headline = list()
    for i in range(VAL_SIZE//para_size):
        doc = val_inputs[i*para_size: min((i+1)*para_size, VAL_SIZE)]
        reference = val_summary[i*para_size: min((i+1)*para_size, VAL_SIZE)].to_list()
        ac_prob = agent.model(doc)  # [tf.newaxis, ...]
        #             ac_prob = tf.math.sigmoid(ac) # 转化成概率
        ac = sample_ac(ac_prob, test=True)  # sample_ac 中包含了tf.math.sigmoid()
        for sam_ac in ac.numpy():
            ac_res.append(sam_ac)

        # 这里面的rlagent和self.model不是一个 需要再调整 调整后应该能提速
        preds, atten_weights = get_result(ac, sen_dim, sen_num, doc, env1, decoder_maxlen=decoder_maxlen)  # 注意修改decoder_maxlen
        for ref_ind, generate_sequence in enumerate(preds):
            for ind, sequence in enumerate(generate_sequence):
                if sequence == stop_index:
                    generate_sequence = generate_sequence[:ind]
                    break
            trans_summary = summary_tokenizer.sequences_to_texts(np.expand_dims(generate_sequence.numpy()[1:], 0))[0]
            generate_headline.append(trans_summary)
            if len(trans_summary) == 0:  # 没生成文章
                r1, r2, rl = 0, 0, 0
            else:
                r1, r2, rl = rouge_score(trans_summary, reference[ref_ind])
            rouge_res_rl.append([r1, r2, rl])

    RLdf = pd.DataFrame(rouge_res_rl, columns=['r1f', 'r2f', 'rlf'])
    RLdf['reward'] = RLdf['r1f'] + RLdf['r2f'] + RLdf['rlf']  # + RLdf['rlf']
    RLdf['rew_new'] = RLdf['reward']-base_reward1[0]
    RLdf.to_csv("/share/home/xuaobo/ai_studio/tasks/res/Chinese/rl_val4/experiment_ep{}_{}.csv".format(ep_num, n_exp))
    ACdf = pd.DataFrame(ac_res, columns=np.linspace(1, sen_num, num=sen_num))
    ACdf.to_csv("/share/home/xuaobo/ai_studio/tasks/res/Chinese/rl_val4/action_ep{}_{}.csv".format(ep_num, n_exp))
    genHead_df = pd.DataFrame(generate_headline, columns=["generate headline"])
    genHead_df.to_csv("/share/home/xuaobo/ai_studio/tasks/res/Chinese/rl_val4/generate_headline_ep{}_{}.csv".format(
        ep_num, n_exp))
    val_mean = np.mean(RLdf['rew_new'])
    wandb.log({"val reward:": val_mean})

    RLdf["reward_1f"] = RLdf["r1f"] - base_r1
    RLdf["reward_2f"] = RLdf["r2f"] - base_r2
    RLdf["reward_lf"] = RLdf["rlf"] - base_r3
    wandb.log({"reward_1f": np.mean(RLdf["reward_1f"])})
    wandb.log({"reward_2f": np.mean(RLdf["reward_2f"])})
    wandb.log({"reward_lf": np.mean(RLdf["reward_lf"])})
    # print((time.time() - start) / 60, 'min')


for i in tqdm(range(n_experiment)):
    start = time.time()
    n_iter = 5
    # 后面不用随机的时候注意注释掉，这时候n_iter会替换成代码开始定义的n_iter
    # 随机的话随机性太大了，不同数据训练的次数不一样，相当于做了加权
    train_PG(n_iter, batch_size_rl, n_exp=i)
