'''
    Ebusiness 数字化
    基于词
'''
import tensorflow as tf
import numpy as np
import sys
import jieba
import pandas as pd
from sklearn.utils import shuffle
import copy

from utils import load_vocab


def toid_Punc():
    """
    在句子中随机添加标点符号以达到数据增强的目的
    :return:
    """

    m_samples_train_s = 0
    m_samples_train_s_pos = 0
    m_samples_train_s_neg = 0

    m_samples_train_l = 0
    m_samples_train_l_pos = 0
    m_samples_train_l_neg = 0

    m_samples_val = 0
    m_samples_val_pos = 0
    m_samples_val_neg = 0

    train_s_writer = tf.io.TFRecordWriter('data/TFRecordFile/train_ps_word.tfrecord')
    train_l_writer = tf.io.TFRecordWriter('data/TFRecordFile/train_pl_word.tfrecord')
    val_writer = tf.io.TFRecordWriter('data/TFRecordFile/val_p_word.tfrecord')

    word_dict = load_vocab("data/OriginalFile/word_dict.txt")

    punc = [word_dict["，"], word_dict["？"], word_dict["！"], word_dict["："], word_dict["；"]]

    data = pd.read_csv("data/OriginalFile/Ebusiness.csv")

    data = shuffle(data)

    k = 1

    for index, row in data.iterrows():
        sen = row["evaluation"].lower().strip()
        label = row["label"].strip()

        if label == "正面":
            label = 1
        else:
            label = 0

        print(k)
        print("sen: ", sen)
        print("lab: ", label)
        print()

        sen = jieba.lcut(sen)

        sen2id = [word_dict[word] if word in word_dict.keys() else word_dict["[UNK]"] for word in sen]
        sen_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in
                       sen2id]

        label_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))

        seq_example = tf.train.SequenceExample(
            feature_lists=tf.train.FeatureLists(feature_list={
                'sen': tf.train.FeatureList(feature=sen_feature),
            }),
            context=tf.train.Features(feature={
                'label': label_feature
            }),

        )

        serialized = seq_example.SerializeToString()

        if np.random.rand() < 0.1:
            val_writer.write(serialized)
            m_samples_val += 1
            if label == 1:
                m_samples_val_pos += 1
            else:
                m_samples_val_neg += 1
        else:
            train_s_writer.write(serialized)
            train_l_writer.write(serialized)
            m_samples_train_s += 1
            m_samples_train_l += 1
            if label == 1:
                m_samples_train_s_pos += 1
                m_samples_train_l_pos += 1
            else:
                m_samples_train_s_neg += 1
                m_samples_train_l_neg += 1

            if np.random.rand() < 0.3:
                sen2idnew = copy.copy(sen2id)

                for idx in reversed(range(len(sen2id))):
                    if np.random.rand() < 0.15:
                        sen2idnew.insert(idx, np.random.choice(punc))

                if len(sen2idnew) != len(sen2id):
                    sen_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in
                                   sen2idnew]
                    seq_example = tf.train.SequenceExample(
                        feature_lists=tf.train.FeatureLists(feature_list={
                            'sen': tf.train.FeatureList(feature=sen_feature),
                        }),
                        context=tf.train.Features(feature={
                            'label': label_feature
                        }),

                    )

                    serialized = seq_example.SerializeToString()
                    train_l_writer.write(serialized)

                    m_samples_train_l += 1
                    if label == 1:
                        m_samples_train_l_pos += 1
                    else:
                        m_samples_train_l_neg += 1

        k += 1

    print('\n')

    print("训练集small样本总量共：%d ,正样本共：%d ,负样本共：%d" % (
        m_samples_train_s, m_samples_train_s_pos, m_samples_train_s_neg))  # 训练样本总量共：3829 ,正样本共：1702 ,负样本共：2127
    print("训练集large样本总量共：%d ,正样本共：%d ,负样本共：%d" % (
        m_samples_train_l, m_samples_train_l_pos, m_samples_train_l_neg))  # 训练样本总量共：3829 ,正样本共：1702 ,负样本共：2127
    print('测试样本总量共：%d ,正样本共：%d ,负样本共：%d ' % (
        m_samples_val, m_samples_val_pos, m_samples_val_neg))  # 测试样本总量共：454 ,正样本共：206 ,负样本共：248


def toid_Repeat():
    """
    在句子中随机重复字词以达到数据增强的目的
    :return:
    """

    m_samples_train_s = 0
    m_samples_train_s_pos = 0
    m_samples_train_s_neg = 0

    m_samples_train_l = 0
    m_samples_train_l_pos = 0
    m_samples_train_l_neg = 0

    m_samples_val = 0
    m_samples_val_pos = 0
    m_samples_val_neg = 0

    train_s_writer = tf.io.TFRecordWriter('data/TFRecordFile/train_rs_word.tfrecord')
    train_l_writer = tf.io.TFRecordWriter('data/TFRecordFile/train_rl_word.tfrecord')
    val_writer = tf.io.TFRecordWriter('data/TFRecordFile/val_r_word.tfrecord')

    word_dict = load_vocab("data/OriginalFile/word_dict.txt")

    data = pd.read_csv("data/OriginalFile/Ebusiness.csv")

    data = shuffle(data)

    k = 1

    for index, row in data.iterrows():
        sen = row["evaluation"].lower().strip()
        label = row["label"].strip()

        if label == "正面":
            label = 1
        else:
            label = 0

        print(k)
        print("sen: ", sen)
        print("lab: ", label)
        print()

        senw = jieba.lcut(sen)

        sen2id = [word_dict[word] if word in word_dict.keys() else word_dict["[UNK]"] for word in senw]
        sen_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in sen2id]

        label_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))

        seq_example = tf.train.SequenceExample(
            feature_lists=tf.train.FeatureLists(feature_list={
                'sen': tf.train.FeatureList(feature=sen_feature),
            }),
            context=tf.train.Features(feature={
                'label': label_feature
            }),

        )

        serialized = seq_example.SerializeToString()

        if np.random.rand() < 0.1:
            val_writer.write(serialized)
            m_samples_val += 1
            if label == 1:
                m_samples_val_pos += 1
            else:
                m_samples_val_neg += 1
        else:
            train_s_writer.write(serialized)
            train_l_writer.write(serialized)
            m_samples_train_s += 1
            m_samples_train_l += 1
            if label == 1:
                m_samples_train_s_pos += 1
                m_samples_train_l_pos += 1
            else:
                m_samples_train_s_neg += 1
                m_samples_train_l_neg += 1

            if np.random.rand() < 0.3:
                sennew = [s for s in sen]

                for idx in reversed(range(len(sen))):
                    if np.random.rand() < 0.15:
                        sennew.insert(idx, sen[idx])

                if len(sennew) != len(sen):
                    print(sen)
                    print("".join(sennew))

                    senw = jieba.lcut("".join(sennew))

                    sen2id = [word_dict[word] if word in word_dict.keys() else word_dict["[UNK]"] for word in senw]
                    sen_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in sen2id]

                    seq_example = tf.train.SequenceExample(
                        feature_lists=tf.train.FeatureLists(feature_list={
                            'sen': tf.train.FeatureList(feature=sen_feature),
                        }),
                        context=tf.train.Features(feature={
                            'label': label_feature
                        }),

                    )

                    serialized = seq_example.SerializeToString()
                    train_l_writer.write(serialized)

                    m_samples_train_l += 1
                    if label == 1:
                        m_samples_train_l_pos += 1
                    else:
                        m_samples_train_l_neg += 1

        k += 1

    print('\n')

    print("训练集small样本总量共：%d ,正样本共：%d ,负样本共：%d" % (
        m_samples_train_s, m_samples_train_s_pos, m_samples_train_s_neg))  # 训练集small样本总量共：3828 ,正样本共：1715 ,负样本共：2113
    print("训练集large样本总量共：%d ,正样本共：%d ,负样本共：%d" % (
        m_samples_train_l, m_samples_train_l_pos, m_samples_train_l_neg))  # 训练集large样本总量共：4862 ,正样本共：2134 ,负样本共：2728
    print('测试样本总量共：%d ,正样本共：%d ,负样本共：%d ' % (
        m_samples_val, m_samples_val_pos, m_samples_val_neg))  # 测试样本总量共：455 ,正样本共：193 ,负样本共：262


if __name__ == '__main__':
    toid_Repeat()
