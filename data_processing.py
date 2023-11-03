import numpy as np
from os import listdir
from os.path import isfile, join
import csv
import pandas as pd
import random







embeddings_dict = {}
with open("./data/glove.6B.50d.txt", 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector
np.save('./data/wordList', np.array(list(embeddings_dict.keys())))
np.save('./data/wordVectors', np.array(list(embeddings_dict.values()), dtype='float32'))

wordList=np.load('./data/wordList.npy')
print('词库导入好')
wordList.tolist()
wordList=[word.encode('UTF-8').decode('UTF-8') for word in wordList]
wordVectors=np.load('./data/wordVectors.npy')
print('词向量加载好')

numfiles=0
maxSeqlength=25
numDimensions=50
data=r"data.csv"
#df = pd.read_csv(data, sep='\t')
#df.to_csv('example.csv', index=False)
output_rows = []
# 读取CSV文件并删除第一行
with open(data, 'r') as file:
    reader = csv.reader(file)
    # 跳过第一行
    next(reader)
    # 读取每一行数据
    for row in reader:
        # 处理第三列和第四列
        if int(row[5]) ==2:
            row[2] = 2  # 第三列设置为1

        elif int(row[5]) == 1:
            row[5] = 1
        else:
            row[5] =0


        new_row = [row[5], row[6]]  # 只留367列
        output_rows.append(new_row)
        numfiles = numfiles + 1


import re
def clean_string(input_string):
    cleaned_string = re.sub(r'@[^\s:]+', '', input_string)
    cleaned_string = re.sub(r'[!@#$%^&*().,";:~\[\]<>?]', '', cleaned_string)
    cleaned_string = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned_string)
    return cleaned_string
#把句子中符号去除

#words = np.zeros((maxSeqlength),dtype='str')
'''ids = np.zeros((numfiles, maxSeqlength), dtype='int32')#
for i in range(numfiles):
    j=0
    output_rows[i][1] = clean_string(output_rows[i][1])
    words = re.findall(r'\b\w+\b', output_rows[i][1])
    if len(words)<maxSeqlength:
        for temp in range(len(words)):
            try:
                ids[i][j] = wordList.index(words[temp])
                j=j+1
            except ValueError:
                ids[i][j] =399999
    else:
        for temp in range(maxSeqlength):
            try:
                ids[i][j] = wordList.index(words[temp])
                j = j + 1
            except ValueError:
                ids[i][j] = 399999
np.save('idsMatrix', ids)'''''
ids = np.load('./idsMatrix.npy')

'''def get_trainbatch():
    labels = []
    arr1 = np.zeros([24, maxSeqlength])
    arr2 = np.zeros([24,maxSeqlength,300])
    for i in range(24):
        num = 12000
        if (output_rows[num]==1):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr1[i] = ids[num:num+1]

    get_trainbatch(i)'''






