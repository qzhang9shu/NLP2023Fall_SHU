import copy
import torch.nn as nn
import numpy as np
import torch

from nltk import word_tokenize
import nltk
nltk.download('punk')

from torch.autograd import Variable

# !pip install rouge_chinese
from rouge_chinese import Rouge
import jieba

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 处理测试数据
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

# def greedy_decode(model, src, src_mask, max_len, start_symbol, top_k=3):
#     memory = model.encode(src, src_mask)
#     ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
#     beam_width = top_k  # 设置束宽度

#     # 初始化束的列表，每个束包含一个序列和对应的分数
#     beams = [(ys, 0)]

#     for _ in range(max_len-1):
#         candidates = []
#         for seq, score in beams:
#             if seq[0, -1] == 2:  # 如果已经生成了终止符，则将其添加到候选列表中
#                 candidates.append((seq, score))
#                 continue
#             out = model.decode(memory, src_mask, seq, None)
#             prob = model.generator(out[:, -1])
#             top_scores, top_indices = torch.topk(prob, beam_width, dim=1)
#             for i in range(beam_width):
#                 next_word = top_indices[0, i].item()
#                 next_score = top_scores[0, i].item()
#                 new_seq = torch.cat([seq, torch.tensor([[next_word]]).type_as(src.data)], dim=1)
#                 candidates.append((new_seq, score + next_score))
        
#         # 从候选列表中选择分数最高的 top_k 个序列作为新的束
#         candidates.sort(key=lambda x: x[1], reverse=True)
#         beams = candidates[:top_k]

#     # 返回得分最高的序列
#     best_seq, best_score = beams[0]
#     return best_seq

def beam_search_decode(model, src, src_mask, max_len, start_symbol, beam_size):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    beam = [{'seq': ys, 'score': 0.0}]
    
    for _ in range(max_len-1):
        candidates = []
        for b in beam:
            out = model.decode(memory, src_mask, b['seq'], Variable(subsequent_mask(b['seq'].size(1)).type_as(src.data)))
            prob = model.generator(out[:, -1])
            top_scores, top_words = torch.topk(prob, beam_size, dim=1)
            
            for score, word in zip(top_scores[0], top_words[0]):
                new_seq = torch.cat([b['seq'], word.view(1, 1)], dim=1)
                new_score = b['score'] + score.item()
                candidates.append({'seq': new_seq, 'score': new_score})
        
        candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        beam = candidates[:beam_size]
    
    return beam[0]['seq']

def get_rouges(pred, ref):
  rouge = Rouge()
  scores = rouge.get_scores(" ".join(list(pred)), " ".join(list(ref)))
  return scores
