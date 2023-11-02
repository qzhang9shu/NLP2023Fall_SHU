import os
import torch
import numpy as np
from nltk import word_tokenize
from collections import Counter
from torch.autograd import Variable
from parser import args
from utils import seq_padding, subsequent_mask

# 保存类用的库
import pickle

import nltk
nltk.download('punkt')

def DataLoader():
    filename = args.prepare_data
    if os.path.exists(filename):
        print("\nLoading Data...")
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        print("\nFind no DataLoader, Preparing...")
        return PrepareData()


class PrepareData:
    def __init__(self):

        # 读取数据 并分词
        self.train_src, self.train_tgt = self.load_data(args.train_file)
        self.test_src, self.test_tgt = self.load_data(args.test_file)
        self.val_src, self.val_tgt = self.load_data(args.val_file)

        # 构建单词表
        self.src_word_dict, self.src_total_words, self.src_index_dict = self.build_dict(self.train_src)
        self.tgt_word_dict, self.tgt_total_words, self.tgt_index_dict = self.build_dict(self.train_tgt)

        # id化
        self.train_src, self.train_tgt = self.wordToID(self.train_src, self.train_tgt, self.src_word_dict, self.tgt_word_dict)
        self.test_src, self.test_tgt = self.wordToID(self.test_src, self.test_tgt, self.src_word_dict, self.tgt_word_dict)
        self.val_src, self.val_tgt = self.wordToID(self.val_src, self.val_tgt, self.src_word_dict, self.tgt_word_dict)
        
        # 划分batch + padding + mask
        self.train_data = self.splitBatch(self.train_src, self.train_tgt, args.batch_size)
        self.test_data = self.splitBatch(self.test_src, self.test_tgt, args.batch_size)
        self.val_data = self.splitBatch(self.val_src, self.val_tgt, args.batch_size)
        
        # 保存这个类对象，不然每次 evaluate/generate 的时候会花更多时间 -> prepare_data.pkl
        self.save_data()

    def save_data(self):
        with open(args.prepare_data, 'wb') as file:
            pickle.dump(self, file)
    
    def load_data(self, path):
        src = []
        tgt = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= args.max_dataset_size:
                    break

                items = line.strip().split('!=!')

                src.append(["BOS"] + word_tokenize(" ".join([w for w in items[1]])) + ["EOS"])
                tgt.append(["BOS"] + word_tokenize(" ".join([w for w in items[0]])) + ["EOS"])

        return src, tgt
    
    def build_dict(self, sentences, max_words = 50000):
        word_count = Counter()

        for sentence in sentences:
            for s in sentence:
                word_count[s] += 1

        ls = word_count.most_common(max_words)
        total_words = len(ls) + 2

        # map 一下
        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
        word_dict['UNK'] = args.UNK
        word_dict['PAD'] = args.PAD

        index_dict = {v: k for k, v in word_dict.items()}

        return word_dict, total_words, index_dict

    def wordToID(self, src, tgt, src_dict, tgt_dict, sort=True):
        length = len(src)

        out_src_ids = [[src_dict.get(w, 0) for w in sent] for sent in src]
        out_tgt_ids = [[tgt_dict.get(w, 0) for w in sent] for sent in tgt]

        # sort sentences by lengths
        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        # 按照同样的顺序排序
        if sort:
            sorted_index = len_argsort(out_src_ids)
            out_src_ids = [out_src_ids[i] for i in sorted_index]
            out_tgt_ids = [out_tgt_ids[i] for i in sorted_index]
            
        return out_src_ids, out_tgt_ids

    # 划分 Batch 并随机打乱一小部分
    def splitBatch(self, src, tgt, batch_size, shuffle=True):
        idx_list = np.arange(0, len(src), batch_size)
        if shuffle:
            np.random.shuffle(idx_list)
        batch_indexs = []
        for idx in idx_list:
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(src))))
        
        batches = []
        for batch_index in batch_indexs:
            batch_src = [src[index] for index in batch_index]  
            batch_tgt = [tgt[index] for index in batch_index]
            batch_tgt = seq_padding(batch_tgt)
            batch_src = seq_padding(batch_src)
            batches.append(Batch(batch_src, batch_tgt))

        return batches


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):

        src = torch.from_numpy(src).to(args.device).long()
        trg = torch.from_numpy(trg).to(args.device).long()

        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask