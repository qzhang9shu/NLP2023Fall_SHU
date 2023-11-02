import torch
import numpy as np
import time

from parser import args
from torch.autograd import Variable
from utils import subsequent_mask, beam_search_decode, get_rouges

def log(data, timestamp):
    file = open(f'log/log-{timestamp}.txt', 'a')
    file.write(data)
    file.write('\n')
    file.close()

def evaluate(data, model, beam_size):
    timestamp = time.time()

    rouge_scores = {'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                    'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                    'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}}

    with torch.no_grad():
        for i in range(len(data.val_src)):
            src_sent = "".join([data.src_index_dict[w] for w in data.val_src[i]])
            src_sent = src_sent[3: -3]
            print("\n" + "Src: " + src_sent)
            log(src_sent, timestamp)

            tgt_sent = "".join([data.tgt_index_dict[w] for w in data.val_tgt[i]])
            tgt_sent = tgt_sent[3: -3]
            print("Tgt: " +  "".join(tgt_sent))
            log(tgt_sent, timestamp)

            src = torch.from_numpy(np.array(data.val_src[i])).long().to(args.device)
            src = src.unsqueeze(0)
            src_mask = (src != 0).unsqueeze(-2)

            out = beam_search_decode(model, src, src_mask, max_len=args.max_length, start_symbol=data.tgt_word_dict["BOS"], beam_size=beam_size)

            predict = []
            for j in range(1, out.size(1)):
                sym = data.tgt_index_dict[out[0, j].item()]
                if sym != 'EOS':
                    predict.append(sym)
                else:
                    break
            print("Pred: %s" % "".join(predict))
            log("Pred: " + "".join(predict) + "\n", timestamp)

            # 计算 Rouge 指标
            scores = get_rouges(predict, tgt_sent)

            # 更新 Rouge 分数累加和
            for rouge_type in rouge_scores:
                for metric in scores[0][rouge_type]:
                    rouge_scores[rouge_type][metric] += scores[0][rouge_type][metric]

          # 计算 Rouge 平均值
        for rouge_type in rouge_scores:
            for metric in rouge_scores[rouge_type]:
                rouge_scores[rouge_type][metric] /= len(data.val_src)

        print("Rouge scores (average):", rouge_scores)


