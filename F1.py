from sklearn.metrics import f1_score

# 读取真实关键词文件
with open('./data/test_trg.txt', 'r', encoding='utf-8') as file:
    true_keywords_list = [line.strip().split() for line in file]

# # 读取预测关键词文件
# with open('./data/sgd_pre.txt', 'r', encoding='utf-8') as file:
#     predicted_keywords_list = [line.strip().split() for line in file]

# 读取预测关键词文件
with open('./data/predicted_keywords.txt', 'r', encoding='utf-8') as file:
    predicted_keywords_list = [line.strip().split() for line in file]

# 初始化F1值列表
f1_values = []

# 遍历每一行文档进行F1值计算
for i, (true_keywords, predicted_keywords) in enumerate(zip(true_keywords_list, predicted_keywords_list)):
    true = set(true_keywords)
    predicted = set(predicted_keywords)
    common = len(true.intersection(predicted))
    precision = common / len(predicted)
    recall = common / len(true)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    f1_values.append(f1)

# 计算平均F1值
average_f1 = sum(f1_values) / len(f1_values)
print("Average F1 Score:", average_f1)
