import torch
import torch.nn as nn
#from hanlp.components.amr.seq2seq import optim
import torch.optim as optim
import torch.nn.functional as F
from model import get_trainbatch,get_testbatch
import torch.nn as nn
import numpy as np
from data_processing import clean_string,wordVectors,wordList
class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # LSTM层
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)

        # 注意力层
        self.attention = nn.Linear(hidden_dim, hidden_dim)

        # 分类层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # LSTM层
        out, (h_n, c_n) = self.lstm(x)

        # 注意力层
        attention_weights = torch.softmax(self.attention(out), dim=1)
        attention_output = torch.sum(attention_weights * out, dim=1)

        # 分类层
        output = self.fc(attention_output)

        return output


def train():
        # load
        input_size = 50
        hidden_size = 64
        num_layers = 1
        dropout=0.3

        '''print('loading...')
        iteration = 2000
        print('training...')
        model = AttentionLSTM(input_dim=input_size, hidden_dim=hidden_size, n_layers=num_layers,dropout=dropout,output_dim=3)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        loss = 0
        for i in range(iteration):
            x,y=get_trainbatch(i)
            # print(y)
            input_ = torch.tensor(x, dtype=torch.float32)
            label = torch.tensor(y, dtype=torch.float32)
            optimizer.zero_grad()
            output = model(input_)
            # print(label)
            # print(output)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            print('iteration:%d loss:%.5f' % (i, loss.item()))
        state = {'model': model.state_dict(),'optimizer': optimizer.state_dict()}
        torch.save(model, 'LSTMatt.pth')
        print('loading...')
        print('testing...')'''
        model=torch.load('LSTMatt.pth')
        model.eval()
        num = 0
        xx,yy=get_testbatch()
        input_ = torch.tensor(xx, dtype=torch.float32)
        label = torch.tensor(yy, dtype=torch.float32)
        output = model(input_)
        pred = output.max(dim=-1)[1]
        label = label.max(dim=-1)[1]
        for k in range(7200):
            if pred[k] == label[k]:
                num += 1
        print('Accuracy：', num / 7200)
def predict(data):
    model = torch.load('LSTMatt.pth')
    model.eval()
    input_ = torch.tensor(data, dtype=torch.float32)
    output = model(input_)
    pred = output.max(dim=-1)[1]
    if pred ==2 :
        a='该句子不为仇恨言论'
        print('该句子不为仇恨言论')
    else:
        a='该句子为仇恨言论'
        print('该句子为仇恨言论')
    return pred,a
def predict2(data):
    model = torch.load('LSTMatt.pth')
    model.eval()
    input_ = torch.tensor(data, dtype=torch.float32)
    output = model(input_)
    pred = output.max(dim=-1)[1]
    return pred
def mask(sentence):
    length=len(sentence)
    flag=0
    sentence_ = []
    for i in range(length-1,1,-1):  #窗长

        for j in range(0,length-i+1): #从头滑动
            sentence_ = []
            for k in range(j,j+i):
                sentence_.append(sentence[k])
            d, _ = sentence_process(sentence_)
            a = predict2(d)
            if a == 2:
                flag=1
                break
        if flag==1:
            break
    return sentence_



def sentence_process(sentence):
    if isinstance(sentence, str):
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
    return arr2,sentence
def begin():
    train()

def begin3(sentence):
    #sentence = "What happen to them vixen ent bitches"" they got ran and threw to the side like a foothill bitch"
    data, sentence = sentence_process(sentence)
    b,a = predict(data)
    sentence2=''
    sentence_=[]
    if b != 2:
        sentence_=mask(sentence)
        for i in range(len(sentence)):
            sentence_.append(0)
        for i in range(len(sentence)):
            if sentence_[i] != sentence[i]:
                sentence_.insert(i, '*')
        for i in range(len(sentence)):
            sentence2=sentence2+sentence_[i]+' '
        return sentence2,a
    else:
        for i in range(len(sentence)):
            sentence2 = sentence2 + sentence[i] + ' '
        return sentence2, a