import torch.nn as nn
import numpy as np
from data_processing import maxSeqlength,ids,output_rows,wordVectors
#from model import BATCH_SIZE,INPUT_SIZE
from random import randint
from model import lstm
from model import get_trainbatch


model=lstm()
# 4. 训练模型
for epoch in range(1):
        nextBatch, nextBatchLabels = get_trainbatch()
        # 获取输入数据和标签

        # 前向传播
        outputs = model(nextBatch)

        # 计算损失
        loss = nn.reduce_mean(nn.softmax_cross_entropy_with_logits(logits=outputs, labels=nextBatchLabels))

        # 反向传播和优化
        nn.optimizer.zero_grad()
        loss.backward()
        optimizer = nn.train.AdamOptimizer().minimize(loss)
