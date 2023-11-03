import random
import numpy as np
from data_processing import maxSeqlength,ids,output_rows,wordVectors
#from model import BATCH_SIZE,INPUT_SIZE
from random import randint
BATCH_SIZE=16
INPUT_SIZE=50
arr1=[]
def get_trainbatch(iteration):
    labels = []
    arr2 = np.zeros([BATCH_SIZE ,maxSeqlength, 50])

    for i in range(BATCH_SIZE):
        if iteration<=1060:
            num = i+BATCH_SIZE*iteration
            arr1.append(num)

        elif iteration>1060 and iteration%1060<1060:
            random.shuffle(arr1)
            num=arr1[i]
        for j in range(maxSeqlength):
            arr2[i][j] = wordVectors[ids[num][j]]
        if (output_rows[num][0] == 1):
            labels.append([1, 0, 0])
        elif (output_rows[num][0] == 0):
            labels.append([0, 1, 0])
        else:
            labels.append([0, 0, 1])
    return arr2, labels
def get_testbatch():
    labels = []
    Setence_number=1
    arr = np.zeros([7200,maxSeqlength, INPUT_SIZE])
    for i in range(7200):
        num = i+17000
        for j in range(maxSeqlength):
            arr[i][j] = wordVectors[ids[num][j]]
        if (output_rows[num][0]==1):
            labels.append([1,0,0])
        elif (output_rows[num][0] == 0):
            labels.append([0,1,0])
        else:
            labels.append([0,0,1])

    return arr, labels
#model=lstm()
# 4. 训练模型
''''for epoch in range(1):
        nextBatch, nextBatchLabels = get_trainbatch()
        # 获取输入数据和标签

        # 前向传播
        outputs = model(torch.tensor(nextBatch, dtype=torch.float32))
        #print(outputs)
        # 将标签转换为 PyTorch 张量
        nextBatchLabels = torch.tensor(nextBatchLabels, dtype=torch.float32)
        print(nextBatchLabels)
        # 计算损失
        loss_fn = nn.CrossEntropyLoss()
        #print(loss)
         #反向传播和优化
         #optimizer.zero_grad()
         #loss.backward()
         #optimizer.step()'''''






