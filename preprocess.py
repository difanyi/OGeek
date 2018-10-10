import pandas as pd
from pyhanlp import *
from gensim.models import Word2Vec, KeyedVectors
import numpy as np


# 词向量长度
DIM = 300


def load_data(train_data_path, val_data_path, test_data_path):
    """
        读取原始数据
    Args:
        train_data_path:
        val_data_path:
        test_data_path:
    Returns:

    """
    columns = ['prefix', 'query_prediction', 'title', 'tag', 'label']
    train_data = pd.read_table(train_data_path, header=None, error_bad_lines=False)
    train_data.columns = columns
    val_data = pd.read_table(val_data_path, header=None, error_bad_lines=False)
    val_data.columns = columns
    test_data = pd.read_table(test_data_path, header=None, error_bad_lines=False)
    test_data.columns = columns[:-1]
    return train_data, val_data, test_data


def tag_onehot(train_data, val_data, test_data):
    """
        tag 向量化
    Args:
        train_data:
        val_data:
        test_data:
    Returns:

    """
    tag = pd.concat([train_data['tag'], val_data['tag'], test_data['tag']])
    onehot = pd.get_dummies(tag)
    onehot.reset_index(drop=True, inplace=True)
    train_tag = onehot.iloc[:train_data.shape[0]]
    val_tag = onehot.iloc[train_data.shape[0]: -test_data.shape[0]]
    test_tag = onehot.iloc[-test_data.shape[0]:]
    val_tag.reset_index(drop=True, inplace=True)
    test_tag.reset_index(drop=True, inplace=True)
    train_data = pd.concat([train_data[train_data.columns[:-2]], train_tag, train_data['label']], axis=1)
    val_data = pd.concat([val_data[val_data.columns[:-2]], val_tag, val_data['label']], axis=1)
    test_data = pd.concat([test_data[test_data.columns[:-1]], test_tag], axis=1)
    print(train_data.columns)
    return train_data, val_data, test_data


def load_word2vec(vec_path):
    """
        加载外部词向量
        参考 https://github.com/Embedding/Chinese-Word-Vectors
    Args:
        vec_path:
    Returns:

    """
    w2c = KeyedVectors.load_word2vec_format(vec_path, binary=False)
    return w2c


def generate_feature(data, word2vec):
    """
        生成特征向量
    Args:
        data: 数据集
        word2vec:
    Returns:

    """
    features = []
    if 'label' in data.columns:
        end = len(data.columns) - 1
    else:
        end = len(data.columns)
    for idx, row in data.iterrows():
        prefix_vec = np.zeros(DIM)
        title_vec = np.zeros(DIM)
        tag_vec = row.iloc[3: end]
        count = 0
        try:
            for word in HanLP.segment(row['prefix']):
                word = str(word).split('/')[0]
                try:
                    prefix_vec += word2vec[word]
                    count += 1
                except:
                    print('word %s not in vocab' % word)
            if count > 0:
                prefix_vec = np.true_divide(prefix_vec, count)
            count = 0
            for word in HanLP.segment(row['title']):
                word = str(word).split('/')[0]
                try:
                    title_vec += word2vec[word]
                    count += 1
                except:
                    print('word %s not in vocab' % word)
            if count > 0:
                title_vec = np.true_divide(title_vec, count)
        except Exception as e:
            print(e)
        feature = np.concatenate((prefix_vec, title_vec, tag_vec))
        features.append(feature)
    return pd.DataFrame(features)


def main():
    train_path = './input/oppo_round1_train_20180929.txt'
    val_path = './input/oppo_round1_vali_20180929.txt'
    test_path = './input/oppo_round1_test_A_20180929.txt'
    vec_path = './input/sgns.merge.word'
    # 加载外部词向量
    word2vec = load_word2vec(vec_path)
    # 加载原始数据
    train_data, val_data, test_data = load_data(train_path, val_path, test_path)
    train_data, val_data, test_data = tag_onehot(train_data, val_data, test_data)
    # 生成特征向量
    X_train = generate_feature(train_data, word2vec)
    y_train = train_data['label']
    X_val = generate_feature(val_data, word2vec)
    y_val = val_data['label']
    X_test = generate_feature(test_data, word2vec)

    # 保存处理后的数据集
    data = dict(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test
    )
    for k, v in data.items():
        print(k, v.shape)
    np.savez('./output/data_process.npz', **data)


if __name__ == '__main__':
    main()



