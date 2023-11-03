import jieba
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import re,json,string,pickle
from test import load_model,stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

model_path = "path_to_save_weights.h5"
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('model_config.json', 'r') as f:
    config = json.load(f)
VOCAB_SIZE = config['VOCAB_SIZE']
EMBEDDING_OUT_DIM = config['EMBEDDING_OUT_DIM']
HIDDEN_UNITS = config['HIDDEN_UNITS']
MAX_LEN = config['MAX_LEN']
NUM_CLASSES = config['NUM_CLASSES']

def is_chinese(text):

    if re.search("[\u4e00-\u9FFF]", text):
        return True
    return False

def extract_keywords_with_tfidf(text, num_keywords=8, stopwords=None):
    if stopwords is None:
        stopwords = set()  # 你可以在这里定义一个停用词列表

    # 从data_train.tsv读取数据
    data = pd.read_csv('data_train.tsv')
    segmented_texts = data['text'].tolist()

    # 对输入的text进行jieba分词，并添加到segmented_texts列表中
    segmented_text = " ".join(jieba.cut(text))
    segmented_texts.append(segmented_text)

    # 使用TfidfVectorizer计算TF-IDF
    vectorizer = TfidfVectorizer(max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(segmented_texts)

    # 获取单词及其TF-IDF得分
    words = vectorizer.get_feature_names_out()
    tfidf_vector = tfidf_matrix.toarray()[-1]  # 获取最后一个，即text的TF-IDF向量

    # 提取text的关键词
    extracted_keywords = [
        word for idx, word in enumerate(words)
        if word not in stopwords
           and word not in string.punctuation
           and len(word) > 1
    ]
    extracted_keywords = sorted(extracted_keywords, key=lambda x: tfidf_vector[words.tolist().index(x)], reverse=True)[
                         :num_keywords]

    return extracted_keywords

def extract_combined_keywords_from_text(text, model_path="path_to_save_weights.h5", tokenizer =tokenizer,max_len=MAX_LEN, num_keywords=10):
    """
    Extract keywords from given text using the trained model and TF-IDF, then combine results.
    """
    if not is_chinese(text):
        raise ValueError("The provided text is not in Chinese.")

    # Extract keywords using model
    model_keywords = extract_keywords_using_model(text, model_path, tokenizer, max_len)
    # Extract keywords using TF-IDF
    tfidf_keywords_list = extract_keywords_with_tfidf(text, num_keywords)

    # 合并两个列表并去重
    combined_keywords = list(set(model_keywords+tfidf_keywords_list))
    tfidf_keywords_list2 = extract_keywords_with_tfidf(text, 6)
    common_predicted_keywords = [word for word in combined_keywords if word in model_keywords and word in tfidf_keywords_list]
    for word in tfidf_keywords_list[0:3]:
        if word not in common_predicted_keywords:
            common_predicted_keywords.append(word)

    return model_keywords,tfidf_keywords_list,common_predicted_keywords

def model_predicted_keywords(text, prediction):
    predicted_keywords = []
    keyword = ""
    for i, word in enumerate(text):
        if word in stopwords or word in string.punctuation or len(word)<=1:
            continue
        if prediction[i] == 0:  # If B
            if keyword not in predicted_keywords:  # If there's already a keyword being constructed
                predicted_keywords.append(keyword)
                keyword = ""
            keyword = word
        elif prediction[i] == 1:  # If I
            keyword += word
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
    return predicted_keywords

def extract_keywords_using_model(text, model_path="path_to_save_weights.h5", tokenizer =tokenizer,max_len=MAX_LEN):
    """
    Extract keywords from given text using the trained model.
    """
    text = " ".join(jieba.cut(text))
    test_texts = text.split()
    # 使用训练时的tokenizer和label_encoder转换数据
    test_texts_encoded = tokenizer.texts_to_sequences([test_texts])
    # 填充序列
    test_texts_encoded_padded = pad_sequences(test_texts_encoded, maxlen=max_len, padding='post')
    # 2. 使用模型进行预测
    model = load_model(model_path)
    prediction = model.predict(test_texts_encoded_padded)[0]

    return model_predicted_keywords(test_texts,prediction)

# text = "建立现代图书馆伦理精神是对社会理性的诉求,人成为精神价值体现的中心.后现代主义对现代社会文化、价值进行了批判,是超越理性的价值评判尺度.图书馆学理论应在传统的基础上,进行非理性的批判,坚持后现代图书馆的执着,实现超越,通过后现代图书馆学的理论建构,形成现代与后现代传承与发展的理论创新态势. 继进趋势的反映:图书馆现代性浪潮中的后现代性"
#
# model_keywords,tfidf_keywords,combined_keywords=extract_combined_keywords_from_text(text)
# print('lstm预测关键词:',model_keywords)
# print('tfidf预测关键词:', tfidf_keywords )
# print('合并关键词:',combined_keywords)
