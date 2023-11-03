import torch
from hanlp.components.amr.seq2seq import optim
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
from model import get_trainbatch,get_testbatch
import torch.nn as nn
import numpy as np
from data_processing import clean_string,wordVectors,wordList


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def train(arr_train,labels_train,arr_test,labels_test):
    # load
    input_size = 50
    hidden_size = 64
    num_layers = 1
    output_size =2
    print('loading...')
    epoch_num = 1000
    print('training...')
    model = LSTM(input_size, hidden_size, num_layers, output_size)
    optimizer = optim.Adam(model.parameters(), lr=0.000005)
    criterion = nn.CrossEntropyLoss()
    loss = 0
    for i in range(epoch_num):
        x = arr_train
        y = labels_train
            # print(y)
        input_ = torch.tensor(x, dtype=torch.float32)
        label = torch.tensor(y, dtype=torch.float32)
        optimizer.zero_grad()
        output = model(input_)
        #print(label)
    # print(output)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        print('epoch:%d loss:%.5f' % (i, loss.item()))
    # save model
        state = {'model': model.state_dict(),'optimizer': optimizer.state_dict()}
        torch.save(model, 'LSTM.pth')
    print('loading...')
    print('testing...')
    model=torch.load('LSTM.pth')
    model.eval()
    num = 0
    xx = arr_test
    yy = labels_test
    input_ = torch.tensor(xx, dtype=torch.float32)
    print(input_.shape)
    label = torch.tensor(yy, dtype=torch.float32)
    output = model(input_)
    pred = output.max(dim=-1)[1]
    label = label.max(dim=-1)[1]
    print(pred)
    print(label)
    for k in range(10000):
        if pred[k] == label[k]:
            num += 1
    print('Accuracy：', num / 10000)
def predict(sentence):
    model = torch.load('LSTM.pth')
    model.eval()
    input_ = torch.tensor(sentence, dtype=torch.float32)
    output = model(input_)
    print(output)
    pred = output.max(dim=-1)[1]
    print(pred)
    if pred ==1 or pred==2 :
        print('该句子为仇恨言论')
    else :
        print('该句子不为仇恨言论')


def sentence_process(sentence):
    sentence=clean_string(sentence)
    sentence=sentence.split()
    j = 0
    arr2=np.zeros((1,len(sentence),50))
    for i in sentence:
        try:
            arr2[0][j] = wordVectors[wordList.index(i)]
            j=j+1
        except ValueError:
            arr2[0][j] = wordVectors[39999]
    return arr2


arr_train,labels_train=get_trainbatch()
arr_test,labels_test=get_testbatch()
train(arr_train,labels_train,arr_test,labels_test)
#test(arr_test,labels_test)
sentence='we must kill all bitch'
sentence=sentence_process(sentence)
predict(sentence)
