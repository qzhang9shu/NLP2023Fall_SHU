"""
作者: lingengyuan
时间: 2023年 10月 19日
"""
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

max_length = 128
# 读取数据集
train_data = pd.read_csv('/Users/lingengyuan/COLDataset/COLDataset/train.csv')
dev_data = pd.read_csv('/Users/lingengyuan/COLDataset/COLDataset/dev.csv')
test_data = pd.read_csv('/Users/lingengyuan/COLDataset/COLDataset/test.csv')

# 数据预处理
train_texts = train_data['TEXT']
train_labels = train_data['label']
dev_texts = dev_data['TEXT']
dev_labels = dev_data['label']
test_texts = test_data['TEXT']
test_labels = test_data['label']

model_name = "/Users/lingengyuan/Desktop/attactive_test/bert-base-chinese"  # 选择一个合适的BERT模型
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length,
                                return_tensors='pt')
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(label)
        }


# 创建数据集和DataLoader
train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length)
train_data_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)
test_data_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# 定义训练循环
def train_model(model, train_data_loader, num_epochs, learning_rate):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        with tqdm(train_data_loader, unit="batch") as tepoch:
            for batch in tepoch:
                optimizer.zero_grad()
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=total_loss / len(train_data_loader))

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_data_loader)}")

    # 保存训练好的模型
    model_path = 'mymodel.pth'
    torch.save(model.state_dict(), model_path)


# 定义评估函数
def evaluate_model(model, test_data_loader):
    model.eval()
    correct_predictions = 0
    total_samples = 0
    all_predicted_labels = []
    all_true_labels = []

    with torch.no_grad():
        for batch in test_data_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            predicted_labels = torch.argmax(logits, dim=1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

            all_predicted_labels.extend(predicted_labels.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_true_labels, all_predicted_labels)
    precision = precision_score(all_true_labels, all_predicted_labels)
    recall = recall_score(all_true_labels, all_predicted_labels)
    f1 = f1_score(all_true_labels, all_predicted_labels)

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

def predict_text(text, model, tokenizer, max_length):
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1)

    return predicted_label.item()

# 训练模型
num_epochs = 3
learning_rate = 2e-5
train_model(model, train_data_loader, num_epochs, learning_rate)

# 评估模型
evaluate_model(model, test_data_loader)

# 进行预测
text_to_predict = "你是个好人"
predicted_label = predict_text(text_to_predict, model, tokenizer, max_length)
print("Predicted Label:", predicted_label)
