# -*- coding: utf-8 -*-
import jieba
from jieba import analyse
import math
import jieba.posseg as psg
import functools
import time
from sklearn.linear_model import SGDClassifier
import joblib


# 停用词表加载方法
def get_stopword_list():
    stop_word_path = './data/stopword.txt'
    stopword_list = [sw.replace('\n', '') for sw in open(stop_word_path,encoding='utf-8').readlines()]
    return stopword_list


# 分词方法，调用结巴接口
def seg_to_list(sentence, pos=False):
    if not pos:
        seg_list = sentence.strip().replace(';',' ').split()
    else:
        seg_list = psg.cut(sentence)
    return seg_list


# 去除干扰词
def word_filter(seg_list, pos=False):
    stopword_list = get_stopword_list()
    filter_list = []
    for seg in seg_list:
        if not pos:
            word = seg
            flag = 'n'
        else:
            word = seg.word
            flag = seg.flag
        if not flag.startswith('n'):
            continue
        if not word in stopword_list:
            filter_list.append(word)

    return filter_list


# 数据加载，pos为是否词性标注的参数，corpus_path为数据集路径
def load_data(pos=False, corpus_path='./data/train_src.txt'):
    doc_list = []
    inputs = open(corpus_path, 'r', encoding='utf-8')
    for line in inputs:
        seg_list = seg_to_list(line, pos)
        filter_list = word_filter(seg_list, pos)
        doc_list.append(filter_list)
    inputs.close()
    return doc_list


# idf值统计方法
def train_idf(doc_list):
    idf_dic = {}
    tt_count = len(doc_list)

    # 每个词出现的文档数
    for doc in doc_list:
        for word in set(doc):
            idf_dic[word] = idf_dic.get(word, 0.0) + 1.0

    # 按公式转换为idf值，分母加1进行平滑处理
    for k, v in idf_dic.items():
        idf_dic[k] = math.log(tt_count / (1.0 + v))

    # 对于没有在字典中的词，默认其仅在一个文档出现，得到默认idf值
    default_idf = math.log(tt_count / (1.0))
    return idf_dic, default_idf


import numpy as np
#  排序函数，用于topK关键词的按值排序
def cmp(e1, e2):
    res = np.sign(e1[1] - e2[1])
    if res != 0:
        return res
    else:
        a = e1[0] + e2[0]
        b = e2[0] + e1[0]
        if a > b:
            return 1
        elif a == b:
            return 0
        else:
            return -1


# TF-IDF类
class TfIdf(object):
    # 四个参数分别是：训练好的idf字典，默认idf值，处理后的待提取文本，关键词数量
    def __init__(self, idf_dic, default_idf, word_list, keyword_num):
        self.word_list = word_list
        self.idf_dic, self.default_idf = idf_dic, default_idf
        self.tf_dic = self.get_tf_dic()
        self.keyword_num = keyword_num

    # 统计tf值
    def get_tf_dic(self):
        tf_dic = {}
        for word in self.word_list:
            tf_dic[word] = tf_dic.get(word, 0.0) + 1.0

        tt_count = len(self.word_list)
        for k, v in tf_dic.items():
            tf_dic[k] = float(v) / tt_count

        return tf_dic

    # 按公式计算tf-idf
    def get_tfidf(self):
        tfidf_dic = {}
        for word in self.word_list:
            idf = self.idf_dic.get(word, self.default_idf)
            tf = self.tf_dic.get(word, 0)

            tfidf = tf * idf
            tfidf_dic[word] = tfidf

        return self.tf_dic , tfidf_dic


def tfidf_extract(word_list, pos=False, train=True, keyword_num=3):
    if train:
        idf_dic, default_idf = train_idf(doc_list_train_src)
    else:
        idf_dic, default_idf = train_idf(doc_list_test_src)
    tfidf_model = TfIdf(idf_dic, default_idf, word_list, keyword_num)
    tf_list , tfidf_list = tfidf_model.get_tfidf()
    return tf_list , tfidf_list


def keyword_list(keypath='./data/test_trg.txt'):
    inputs = open(keypath, 'r', encoding='utf-8')
    keyword_list =[]
    for line in inputs:
        keyword_list.append(line.strip().replace(';',' ').split())
    inputs.close()
    return keyword_list


def get_first_three_keywords(keywordslist, row=1):
    length = len(keywordslist[row - 1])
    first_word = keywordslist[row - 1][0]
    m = min(length, 3)
    first_three_keywords = set(keywordslist[row - 1][:m])
    return first_word, first_three_keywords


def get_feature_label(readpath='./data/train_src.txt' ,
                      keypath='./data/train_trg.txt',train=True, end =1):
    if train:
        line_length = len(doc_list_train_src)
        doc_list = doc_list_train_src
    else:
        line_length = len(doc_list_test_src)
        doc_list = doc_list_test_src
    keywords = keyword_list(keypath)
    features = []
    target = []
    word = []
    for n in range(line_length):
        print(str(n+1))
        tf_list, tfidf_list = tfidf_extract(doc_list[n], train=train)
        p = 1
        for k, v in tf_list.items():
            temp = []
            if k in keywords[n]:
                target.append(1)
            else:
                target.append(0)
            temp.append(v)
            temp.append(tfidf_list.get(k, 0))
            temp.append(p)
            p = p + 1
            features.append(temp)
        word.append(list(tf_list.keys()))
        if (n+1) == end:
            break
    features = np.array(features)
    labels = np.array(target)
    return features , labels , word


def SGD(classifier, train_features, train_labels,
        test_word ,test_features, test_labels,
        load=True, save=True,
        savepath='./result/sgd.txt', end =1):
    if not load:
        classifier.fit(train_features , train_labels) # 训练模型
        print("model start to save")
        joblib.dump(classifier, 'saved_model/sgd_default.pkl')
        print("model save successfully")

    print("model start to load")
    classifier = joblib.load('saved_model/sgd_best.pkl') # 加载训练好的分类器
    print("model load successfully")

    line_length = len(doc_list_test_src) # 测试集总行数
    c = 0 # 每句的开始序号
    keywordslist = keyword_list() #关键词列表
    string = ''
    for n in range(line_length):
        length = len(test_word[n])
        if length == 0:
            continue
        pred_proba = classifier.predict_proba(test_features[c:(c + length)])
        l = []
        for i in range(min(5,length)):
            t = []
            t.append(test_word[n][i])
            t.append(pred_proba[i][1])
            t.append(test_labels[c+i])
            l.append(t)
        c = c + length
        l.sort(key=functools.cmp_to_key(cmp), reverse=True)
        first_keyword_pred = l[0][0]
        first_three_keywords_pred = set()
        for k, v, p in l[:min(4,length)]: #3变成4
            first_three_keywords_pred.add(k)
        first_keyword , first_three_keywords = get_first_three_keywords(keywordslist, n+1)
        str2 = str(first_three_keywords_pred) + '\n'
        string = string + str2
        if (n+1) == end:
            break
    print('SGD-tfidf model result :')
    if save:  # 是否保存结果
        outputs = open(savepath, 'w', encoding='utf-8')
        outputs.write('预测值' + '\n')
        outputs.write(string)


def main():
    train_features, train_labels, train_word = get_feature_label(end=0) # 训练集特征提取
    test_features, test_labels, test_word = get_feature_label(readpath='./data/test_src.txt',
                                                              keypath='./data/test_trg.txt',
                                                              train=False, end=0) # 测试集特征提取

    print("Numpy start to save")
    np.savez('./data/dataset_own.npz', train_features, train_labels, test_word,
             test_features, test_labels, allow_pickle=True, fix_imports=True) # 保存特征为numpy矩阵
    print("Numpy save successfully")

    print("Numpy start to load")
    npzfile = np.load('./data/dataset_own.npz', allow_pickle=True, fix_imports=True) # 加载特征numpy矩阵
    print("Numpy load successfully")

    # 创建SGD分类器模型
    sgd = SGDClassifier(loss='log', random_state=66)  # 随机数66
    SGD(classifier=sgd,  # 分类器
        train_features=npzfile['arr_0'],  # 训练集特征矩阵形式
        train_labels=npzfile['arr_1'],  # 训练集标签
        test_word=npzfile['arr_2'],  # 测试集词列表，主要用于结果输出
        test_features=npzfile['arr_3'],  # 测试集特征矩阵形式
        test_labels=npzfile['arr_4'],  # 测试集标签矩阵形式
        load=False,  # 是否加载训练好的模型，否的话先训练并保存然后加载模型测试
        save=True,  # 是否保存结果
        savepath='./result/sgd.txt',  # SGD结果保存路径
        end=0)  # 运行至第几行，设置0时为全部


if __name__ == '__main__':
    print(time.strftime('Strat to prepare:%H:%M:%S', time.localtime(time.time())))
    # 预处理加载好训练集和关键词
    doc_list_train_src = load_data(corpus_path='data/train_src.txt')
    doc_list_test_src = load_data(corpus_path='data/test_src.txt')
    print(time.strftime('Preparations have been done:%H:%M:%S', time.localtime(time.time())))
    main()
    print(time.strftime('End time:%H:%M:%S', time.localtime(time.time())))
