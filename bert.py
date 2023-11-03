import torch
from datasets import load_dataset  # hugging-face dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.optim as optim
from torch.nn.functional import one_hot
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torchmetrics.functional import accuracy, recall, precision, f1_score  # lightning中的评估
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import ahocorasick

batch_size = 128
epochs = 10  # 刚才写的30
dropout = 0.4
rnn_hidden = 768
rnn_layer = 1
class_num = 3
lr = 0.001
# todo：自定义数据集
class MydataSet(Dataset):
    def __init__(self, path, split):
        self.dataset = load_dataset('csv', data_files=path, split=split)

    def __getitem__(self, item):
        text = self.dataset[item]['TEXT']
        label = self.dataset[item]['label']
        return text, label

    def __len__(self):
        return len(self.dataset)

token = BertTokenizer.from_pretrained('bert-base-chinese')
# todo: 定义批处理函数
def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    # 分词并编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,  # 单个句子参与编码
        truncation=True,  # 当句子长度大于max_length时,截断
        padding='max_length',  # 一律补pad到max_length长度
        max_length=200,
        return_tensors='pt',  # 以pytorch的形式返回，可取值tf,pt,np,默认为返回list
        return_length=True,
    )

    # input_ids:编码之后的数字
    # attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']  # input_ids 就是编码后的词
    attention_mask = data['attention_mask']  # pad的位置是0,其他位置是1
    token_type_ids = data['token_type_ids']  # (如果是一对句子)第一个句子和特殊符号的位置是0,第二个句子的位置是1
    labels = torch.LongTensor(labels)  # 该批次的labels

    # print(data['length'], data['length'].max())
    return input_ids, attention_mask, token_type_ids, labels


# todo: 定义模型，上游使用bert预训练，下游任务选择双向LSTM模型，最后加一个全连接层
class BiLSTMClassifier(nn.Module):
    def __init__(self, drop, hidden_dim, output_dim):
        super(BiLSTMClassifier, self).__init__()
        self.drop = drop
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 加载bert中文模型,生成embedding层
        self.embedding = BertModel.from_pretrained('bert-base-chinese')
        # 去掉移至gpu
        # 冻结上游模型参数(不进行预训练模型参数学习)
        for param in self.embedding.parameters():
            param.requires_grad_(False)
        # 生成下游RNN层以及全连接层
        self.lstm = nn.LSTM(input_size=768, hidden_size=self.hidden_dim, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=self.drop)
        self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)
        # 使用CrossEntropyLoss作为损失函数时，不需要激活。因为实际上CrossEntropyLoss将softmax-log-NLLLoss一并实现的。

    def forward(self, input_ids, attention_mask, token_type_ids):
        embedded = self.embedding(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embedded = embedded.last_hidden_state  # 第0维才是我们需要的embedding,embedding.last_hidden_state = embedding[0]
        out, (h_n, c_n) = self.lstm(embedded)
        output = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        output = self.fc(output)
        return output


# todo: 定义pytorch lightning
class BiLSTMLighting(pl.LightningModule):
    def __init__(self, drop, hidden_dim, output_dim):
        super(BiLSTMLighting, self).__init__()
        self.model = BiLSTMClassifier(drop, hidden_dim, output_dim)  # 设置model
        self.criterion = nn.CrossEntropyLoss()  # 设置损失函数
        self.train_dataset = MydataSet('train_clean.csv', 'train')
        self.val_dataset = MydataSet('val_clean.csv', 'train')
        self.test_dataset = MydataSet('test_clean.csv', 'train')

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        return optimizer

    def forward(self, input_ids, attention_mask, token_type_ids):  # forward(self,x)
        return self.model(input_ids, attention_mask, token_type_ids)

    def train_dataloader(self):
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                  shuffle=True)
        return train_loader

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch  # x, y = batch
        y = one_hot(labels, num_classes=3)
        # 将one_hot_labels类型转换成float
        y = y.to(dtype=torch.float)
        # forward pass
        y_hat = self.model(input_ids, attention_mask, token_type_ids)
        y_hat = y_hat.squeeze()  # 将[128, 1, 3]挤压为[128,3]
        loss = self.criterion(y_hat, y)  # criterion(input, target)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)  # 将loss输出在控制台
        return loss  # 必须把log返回回去才有用

    def val_dataloader(self):
        val_loader = DataLoader(dataset=self.val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        return val_loader

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        y = one_hot(labels, num_classes=3)
        y = y.to(dtype=torch.float)
        # forward pass
        y_hat = self.model(input_ids, attention_mask, token_type_ids)
        y_hat = y_hat.squeeze()
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return loss

    def test_dataloader(self):
        test_loader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        return test_loader

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        target = labels  # 用于待会儿计算acc和f1-score
        y = one_hot(target, num_classes=3)
        y = y.to(dtype=torch.float)
        # forward pass
        y_hat = self.model(input_ids, attention_mask, token_type_ids)
        y_hat = y_hat.squeeze()
        pred = torch.argmax(y_hat, dim=1)
        acc = (pred == target).float().mean()

        loss = self.criterion(y_hat, y)
        self.log('loss', loss)
        # task: Literal["binary", "multiclass", "multilabel"],对应[二分类，多分类，多标签]
        #  average=None分别输出各个类别, 不加默认算平均
        re = recall(pred, target, task="multiclass", num_classes=class_num, average=None)
        pre = precision(pred, target, task="multiclass", num_classes=class_num, average=None)
        f1 = f1_score(pred, target, task="multiclass", num_classes=class_num, average=None)

        def log_score(name, scores):
            for i, score_class in enumerate(scores):
                self.log(f"{name}_class{i}", score_class)

        log_score("recall", re)
        log_score("precision", pre)
        log_score("f1", f1)
        self.log('acc', accuracy(pred, target, task="multiclass", num_classes=class_num))
        self.log('avg_recall', recall(pred, target, task="multiclass", num_classes=class_num, average="weighted"))
        self.log('avg_precision', precision(pred, target, task="multiclass", num_classes=class_num, average="weighted"))
        self.log('avg_f1', f1_score(pred, target, task="multiclass", num_classes=class_num, average="weighted"))

if __name__ == '__main__':
    token = BertTokenizer.from_pretrained('bert-base-chinese')
    # Trainer可以帮助调试，比如快速运行、只使用一小部分数据进行测试、完整性检查等，
    # 详情请见官方文档https://lightning.ai/docs/pytorch/latest/debug/debugging_basic.html
    # auto自适应gpu数量
    trainer = Trainer(max_epochs=epochs, log_every_n_steps=10, accelerator='cpu', devices="auto",fast_dev_run=False)
    #model = BiLSTMLighting(drop=dropout, hidden_dim=rnn_hidden, output_dim=class_num)
    #trainer.fit(model)

'''def test():
    # 加载之前训练好的最优模型参数
    model = BiLSTMLighting.load_from_checkpoint(checkpoint_path='./content/lightning_logs/version_0/checkpoints/epoch=9-step=1210.ckpt',
                                                drop=dropout, hidden_dim=rnn_hidden, output_dim=class_num)
    trainer = Trainer(fast_dev_run=False)
    result = trainer.test(model)
    print(result)
test()'''
def replace_nth_char(input_string, n, new_char):
    if n < 0 or n >= len(input_string):
        return input_string  # 如果n超出字符串长度范围，不做替换
    else:
        return input_string[:n] + new_char + input_string[n + 1:]
x=[]
def predict(data1, data2, data3,data):
    model = BiLSTMLighting.load_from_checkpoint(
        checkpoint_path='./epoch=19-step=2420.ckpt', drop=dropout,
        hidden_dim=rnn_hidden, output_dim=class_num)
    model.eval()

    input_1 = torch.tensor(data1, dtype=torch.long)
    input_2 = torch.tensor(data2, dtype=torch.long)
    input_3 = torch.tensor(data3, dtype=torch.long)
    output = model(input_1, input_2, input_3)
    predicted_probabilities = torch.softmax(output, dim=1)
    predicted_labels = torch.argmax(predicted_probabilities, dim=1)
    #pred = output.max(dim=-1)[1]
    #print(predicted_labels)
    #predicted_class = torch.argmax(output, dim=1).item()
    n=0
    for i in range(len(predicted_labels)):
        if predicted_labels[i]==1:
            x.append(i)
            n=n+1
    if n==0:
        #print("此言论不为仇恨言论")
        a="此言论不为仇恨言论"
    else:
        #print("此言论为仇恨言论")
        a="此言论为仇恨言论"
    for m in x:
        data=replace_nth_char(data, m, '*')
    print(data)
    return a

def collate_fn1(data):
    sents = [i[0] for i in data]
    # 分词并编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,  # 单个句子参与编码
        truncation=True,  # 当句子长度大于max_length时,截断
        padding='max_length',  # 一律补pad到max_length长度
        max_length=200,
        return_tensors='pt',  # 以pytorch的形式返回，可取值tf,pt,np,默认为返回list
        return_length=True,
    )

    # input_ids:编码之后的数字
    # attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']  # input_ids 就是编码后的词
    attention_mask = data['attention_mask']  # pad的位置是0,其他位置是1
    token_type_ids = data['token_type_ids']  # (如果是一对句子)第一个句子和特殊符号的位置是0,第二个句子的位置是1  # 该批次的labels

    # print(data['length'], data['length'].max())
    return input_ids, attention_mask, token_type_ids

def build_actree(wordlist):
    actree = ahocorasick.Automaton()
    for index, word in enumerate(wordlist):
        actree.add_word(word, (index, word))
    actree.make_automaton()
    return actree

def begin5(data):
    # print(data)
    a, b, c = collate_fn1(data)
    pre=predict(a, b, c, data)
    wordlist = []
    with open("./sensitive_words.txt", 'r', encoding='utf-8') as file:
        lines = file.read().splitlines()
    for line in lines:
        wordlist.append(line)
    actree = build_actree(wordlist)
    sent_cp = data
    for i in actree.iter(data):
        sent_cp = sent_cp.replace(i[1][1], "**")
    #print("屏蔽结果：", sent_cp)
    return sent_cp,pre




