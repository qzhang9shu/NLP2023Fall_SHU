import argparse

import torch


parser = argparse.ArgumentParser()

parser.add_argument('--train_file', default='data/data1.tsv')
parser.add_argument('--test_file', default='data/data2.tsv')
parser.add_argument('--val_file', default='data/data3.tsv')
parser.add_argument('--prepare_data', default='data/prepare_data.pkl')
parser.add_argument('--loss_history', default='data/loss_history.txt')
parser.add_argument('--sentence', default='', type=str)
parser.add_argument('--max_dataset_size', default=200000)

parser.add_argument('--UNK', default=0, type=int)
parser.add_argument('--PAD', default=1, type=int)

# TODO 常改动参数
parser.add_argument('--type', default='train') # 默认是训练模式, 若传递 "evaluate" 则对 dev数据集进行预测输出
parser.add_argument('--pre2train', default=False)
parser.add_argument('--gpu', default=0, type=int) # gpu 卡号
parser.add_argument('--epochs', default=10, type=int) # 训练轮数
parser.add_argument('--layers', default=6, type=int) # transformer层数
parser.add_argument('--h-num', default=8, type=int) # multihead attention hidden层数
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--d-model', default=256, type=int) 
parser.add_argument('--d-ff', default=1024, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--beam_size', default=3, type=int) # 束搜索宽度
parser.add_argument('--max-length', default=60, type=int)
parser.add_argument('--save_model', default='checkpoint/model.pt') # 模型保存位置

parser.add_argument('--mT5', default='csebuetnlp/mT5_multilingual_XLSum')
parser.add_argument('--mT5_mode', default='mT5_train_test')

parser.add_argument('--load_model', default=None) # 占个坑


args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
args.device = device