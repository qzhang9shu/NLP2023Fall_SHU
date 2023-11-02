# -*- coding:utf-8 -*-
import math
import jieba
import jieba.posseg as psg
# from gensim import corpora, models
from jieba import analyse
import functools
import time


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
        # 进行词性标注的分词方法
        seg_list = psg.cut(sentence)
    return seg_list


# 去除干扰词
def word_filter(seg_list, pos=False):
    stopword_list = get_stopword_list()
    filter_list = []
    # 根据POS参数选择是否词性过滤
    # 不进行词性过滤，则将词性都标记为n，表示全部保留
    for seg in seg_list:
        if not pos:
            word = seg
            flag = 'n'
        else:
            word = seg.word
            flag = seg.flag
        if not flag.startswith('n'):
            continue
        # 过滤停用词表中的词，以及长度为<2的词
        if not word in stopword_list:
            filter_list.append(word)

    return filter_list


# 数据加载，pos为是否词性标注的参数，corpus_path为数据集路径
def load_data(pos=False, corpus_path='./data/test_src.txt'):
    # 调用上面方式对数据集进行处理，处理后的每条数据仅保留非干扰词
    doc_list = []
    for line in open(corpus_path, 'r', encoding='utf-8'):
        seg_list = seg_to_list(line, pos)
        filter_list = word_filter(seg_list, pos)
        doc_list.append(filter_list)

    return doc_list


def test(text, num_keywords=3):
    # 引入jieba中的TextRank
    textrank = analyse.textrank
    # 使用jieba的TextRank算法提取关键词
    keyword = textrank(text, topK=num_keywords)
    return keyword


# idf值统计方法
def train_idf(doc_list):
    idf_dic = {}
    # 总文档数
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


#  排序函数，用于topK关键词的按值排序
def cmp(e1, e2):
    import numpy as np
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

        tfidf_dic.items()

        l = sorted(tfidf_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]
        if len(l)==0:
            first_keyword = ''
        else:
            first_keyword = l[0][0]
        first_three_keywords = set()

        n = 1
        # 根据tf-idf排序，去排名前keyword_num的词作为关键词
        for k, v in l:
            # print(k + "/", end='')
            first_three_keywords.add(k)
            if n == 3:
                break
            n = n + 1
        # print()
        return first_keyword , first_three_keywords, l


def tfidf_extract(idf_dic, default_idf, word_list, keyword_num=3, pos=False):
    tfidf_model = TfIdf(idf_dic, default_idf, word_list, keyword_num)
    first_keyword , first_three_keywords, l = tfidf_model.get_tfidf()
    return first_keyword , first_three_keywords, l


def TFIDF(*, savepath, keyword_num=5, end=2, save=True, doc_list_train_src, doc_list_test_src, keywords_list):
    n = 1 # 当前的行数
    string = '' # 储存预测结果用于存储
    idf_dic, default_idf = train_idf(doc_list_train_src)
    # print(1)
    for text in doc_list_test_src:
        first_keyword_pred, first_three_keywords_pred, l = tfidf_extract(idf_dic, default_idf,
                                                                         text,
                                                                         keyword_num=keyword_num)

        if first_keyword_pred == '':
            n = n + 1
            continue

        ground_truth = []
        for k, v in l:
            if k in keywords_list[n-1]:
                ground_truth.append(1)
            else:
                ground_truth.append(0)


        str2 = str(first_three_keywords_pred) + '\n'
        string = string + str2
        if n == end:
            break
        n = n + 1
    print('TF-IDF model result :')
    if save: # 是否保存结果
        outputs = open(savepath, 'w', encoding='utf-8')
        outputs.write('预测值' + '\n')
        outputs.write(string)
        outputs.close()
        print("result save successfully")


if __name__ == '__main__':
    print(time.strftime('Strat to prepare:%H:%M:%S', time.localtime(time.time())))
    # 预处理加载好训练集和关键词
    doc_list_train_src = load_data(corpus_path='./data/train_src.txt')
    doc_list_test_src = load_data(corpus_path='./data/test_src.txt')
    keywords_list = keyword_list('./data/test_trg.txt')  # 它和doc_list_test_trg区别就是没有去停用词

    TFIDF(savepath = './result/tfidf.txt', # TF-IDF结果保存路径
          keyword_num=5, # 候选词个数
          end = 0,  # end代表运行到第几行，设置为0时表示全部
          save=True,
          doc_list_train_src=doc_list_train_src,
          doc_list_test_src=doc_list_test_src,
          keywords_list=keywords_list) # 是否保存结果
    print(time.strftime('End time:%H:%M:%S', time.localtime(time.time())))
