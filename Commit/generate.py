import torch
import numpy as np
import time

from nltk import word_tokenize

from parser import args
from torch.autograd import Variable
from utils import subsequent_mask, beam_search_decode

def log(data, timestamp):
    file = open(f'log/log-{timestamp}.txt', 'a')
    file.write(data)
    file.write('\n')
    file.close()

def generate(data, model):
    timestamp = time.time()
    beam_size = args.beam_size
    with torch.no_grad():
        sentences = list((args.sentence).split(" "))
        
        # 对每句进行 summarize
        for i in range(len(sentences)):
            sent = sentences[i]
            src = []
            src.append(["BOS"] + word_tokenize(" ".join([w for w in sent])) + ["EOS"])

        for i in range(len(src)):
          # Word Embedding
          ############################################################################

          length = len(src)
          out_src_ids = [[data.src_word_dict.get(w, 0) for w in sent] for sent in src]

          # sort sentences by lengths
          def len_argsort(seq):
              return sorted(range(len(seq)), key=lambda x: len(seq[x]))
          # 按照同样的顺序排序

          sorted_index = len_argsort(out_src_ids)
          out_src_ids = [out_src_ids[i] for i in sorted_index]

          src = out_src_ids

          # print(src)
          # print(type(src))
          ############################################################################

          src_sent = "".join(data.src_index_dict[w] for w in src[i]) # 注意，这里开始我们对每个句子进行处理 “src[i]”
          src_sent = src_sent[3: -3]
          
          print("\n" + "Src: " + src_sent)
          log(src_sent, timestamp)

          sent = torch.from_numpy(np.array(src)).long().to(args.device)
          sent = sent.unsqueeze(0)
          sent_mask = (sent != 0).unsqueeze(-2)
          
          # print("\nsent: ", sent)
          # print("sent_mask: ", sent_mask)

          out = beam_search_decode(model, sent, sent_mask, max_len=args.max_length, start_symbol=data.tgt_word_dict["BOS"], beam_size=beam_size)

          # print("out: ", out)
          # print()

          predict = []
          for j in range(1, out.size(1)):
              sym = data.tgt_index_dict[out[0, j].item()]
              if sym != 'EOS':
                  predict.append(sym)
              else:
                  break
          print("Pred: %s" % "".join(predict))
          log("Pred: " + "".join(predict) + "\n", timestamp)