import pandas as pd
import numpy as np
from run_text2text_csl import generate_labels
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tf2crf import CRF, ModelWithCRFLoss
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Model
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import pickle
import string
from sklearn.feature_extraction.text import TfidfVectorizer

with open('model_config.json', 'r') as f:
    config = json.load(f)

# 使用配置中的参数
VOCAB_SIZE = config['VOCAB_SIZE']
EMBEDDING_OUT_DIM = config['EMBEDDING_OUT_DIM']
HIDDEN_UNITS = config['HIDDEN_UNITS']
MAX_LEN = config['MAX_LEN']
NUM_CLASSES = config['NUM_CLASSES']

with open('hit_stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = set(f.read().splitlines())

deletes=['《','》','-']

def load_model(model_path):
    # 重建模型架构
    inputs = Input(shape=(MAX_LEN,), dtype='int32')
    output = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_OUT_DIM, trainable=True, mask_zero=True)(inputs)
    output = Bidirectional(LSTM(units=HIDDEN_UNITS, return_sequences=True))(output)
    # output = Bidirectional(LSTM(units=HIDDEN_UNITS, return_sequences=True))(output)
    crf = CRF(units=NUM_CLASSES, dtype='float32')
    output = crf(output)
    base_model = Model(inputs, output)

    # 加载权重
    base_model.load_weights(model_path)
    # 如果需要，再包装模型
    return ModelWithCRFLoss(base_model)
# Define a function to generate BIO labels from tokenized text and keywords


def calculate_metrics(true_keywords, predicted_keywords):
    """
    Calculate precision, recall and F1-score.
    """
    # Flatten the lists and convert to set for easier comparison
    from sklearn.metrics import precision_score, recall_score, f1_score
    true_set = set([kw for sublist in true_keywords for kw in sublist])
    predicted_set = set([kw for sublist in predicted_keywords for kw in sublist])

    # Convert to binary labels
    true_labels = [1 if kw in true_set else 0 for kw in true_set.union(predicted_set)]
    predicted_labels = [1 if kw in predicted_set else 0 for kw in true_set.union(predicted_set)]

    # Calculate metrics
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    return precision, recall, f1

def extract_keywords_with_tfidf(data, num_keywords):
    # 使用jieba进行分词
    data['segmented_text'] = data['text']
    # 使用TfidfVectorizer计算TF-IDF
    vectorizer = TfidfVectorizer(max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(data['segmented_text'])

    # 获取单词及其TF-IDF得分
    words = vectorizer.get_feature_names_out()

    # 提取关键词

    data['extracted_keywords'] = [
        [words[idx] for idx in tfidf_vector.argsort()[-num_keywords:][::-1] if words[idx] not in stopwords and words[idx] not in string.punctuation and len(words[idx]) > 1]
        for tfidf_vector in tfidf_matrix.toarray()
    ]
    return data['extracted_keywords'].tolist()


def display_predicted_keywords(data_test, predictions):
    predicted_keywords_list = []
    for text, labels in zip(data_test['text'].apply(lambda x: x.split()).tolist(), predictions):
        predicted_keywords = []
        keyword = ""
        for i, word in enumerate(text):
            if word in stopwords or word in string.punctuation or len(word)<=1:
                continue
            if labels[i] == 0:  # If B
                if keyword not in predicted_keywords:  # If there's already a keyword being constructed
                    predicted_keywords.append(keyword)
                    keyword = ""
                keyword = word
            # elif labels[i] == :  # If I
            #     keyword += word
            else:  # If O
                if keyword not in predicted_keywords:  # If there's already a keyword being constructed
                    predicted_keywords.append(keyword)
                    keyword = ""

        # Add last keyword if it exists
        if keyword not in predicted_keywords:
            predicted_keywords.append(keyword)

        # Filter keywords based on conditions
        predicted_keywords = [word for word in predicted_keywords if
                              word not in stopwords and word not in string.punctuation and len(word) > 1]

        predicted_keywords_list.append(predicted_keywords)

    return predicted_keywords_list

# 使用提供的函数显示预测的关键词


if __name__=="__main__":
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    # 标注
    # 1. 加载和预处理测试集
    data_test = pd.read_csv('data_test.tsv', sep=',', encoding='utf-8')
    data_test['labels'] = data_test.apply(lambda row: generate_labels(row['text'], row['keywords'].split('_')), axis=1)
    #
    test_texts = data_test['text'].apply(lambda x:x.split()).tolist()
    test_labels = data_test['labels'].tolist()

    # 使用训练时的tokenizer和label_encoder转换数据
    test_texts_encoded = tokenizer.texts_to_sequences(test_texts)
    test_labels_encoded = [label_encoder.transform(label_seq) for label_seq in test_labels]

    # 填充序列
    test_texts_encoded_padded = pad_sequences(test_texts_encoded, maxlen=MAX_LEN, padding='post')
    test_labels_encoded_padded = pad_sequences(test_labels_encoded, maxlen=MAX_LEN, padding='post')
    print(test_labels_encoded_padded.shape)
    # 2. 使用模型进行预测
    model=load_model('path_to_save_weights.h5')
    predictions = model.predict(test_texts_encoded_padded)
    np.savetxt('predictions.txt',predictions,fmt='%d')
    #
    true_keywords_list = data_test['keywords'].apply(lambda x: x.split("_")).tolist()
    predicted_keywords_list_net = display_predicted_keywords(data_test, predictions)
    predicted_keywords_list_tfidf = extract_keywords_with_tfidf(data_test, 6)
    # max,imax,jmax,kmax=0,0,0,0
    # for i in range(8):
    #     for j in range(i):
    #         for k in range(i):


    common_predicted_keywords = []

    for words1, words2 in zip(predicted_keywords_list_net, predicted_keywords_list_tfidf):

        combined_keywords = list(set(words1 + words2))
        word_list = [word for word in combined_keywords if word in words1 and word in words2]
        for word in words2[0:3]:
            if word not in word_list:
                word_list.append(word)
        common_predicted_keywords.append(word_list)

    precision_net, recall_net, f1_net = calculate_metrics(true_keywords_list, predicted_keywords_list_net)
    precision_tfidf, recall_tfidf, f1_tfidf = calculate_metrics(true_keywords_list, predicted_keywords_list_tfidf)
    precision_common, recall_common, f1_common = calculate_metrics(true_keywords_list, common_predicted_keywords)

    #             if f1_common>max:
    #                 max,imax,jmax,kmax=f1_common,i,j,k
    # print(imax,jmax,kmax,max)
    print('model: 精确度：',precision_net, '召回率：',recall_net,'F1值：', f1_net)
    print('tfidf: 精确度：', precision_tfidf, '召回率：', recall_tfidf, 'F1值：', f1_tfidf)
    print('common: 精确度：', precision_common, '召回率：', recall_common, 'F1值：', f1_common)

    with open('keywords_output.txt', 'w', encoding='utf-8') as file:
        for text, true_keywords, net_keywords, tfidf_keywords, common_keywords in zip(data_test['text'], true_keywords_list, predicted_keywords_list_net, predicted_keywords_list_tfidf, common_predicted_keywords):
            file.write("Text: " + text + "\n")
            file.write("True Keywords: " + "_".join(true_keywords) + "\n")
            file.write("Predicted Keywords (model): " + "_".join(net_keywords) + "\n")
            file.write("Predicted Keywords (TF-IDF): " + "_".join(tfidf_keywords) + "\n")
            file.write("Common Predicted Keywords: " + "_".join(common_keywords) + "\n\n")