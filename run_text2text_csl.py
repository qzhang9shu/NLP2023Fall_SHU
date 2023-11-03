
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tf2crf import CRF, ModelWithCRFLoss
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Model
import numpy as np
import json
import pickle
from tensorflow.python.keras import layers

def generate_labels(tokenized_text, keywords):
    tokens = tokenized_text.split()
    labels = ["O"] * len(tokens)  # initialize with "O"
    for keyword in keywords:
        keyword_tokens = list(jieba.cut(keyword))
        length = len(keyword_tokens)

        # Try to find the keyword in tokens
        for i in range(len(tokens) - length + 1):  # update the range to consider length of keyword
            if tokens[i:i+length] == keyword_tokens:
                labels[i] = "B"

    return labels

# Correctly split the columns
if __name__=="__main__":
    data_tokenized = pd.read_csv('data_train.tsv', sep=',', encoding='utf-8')

    data_tokenized['keywords'] = data_tokenized['keywords'].apply(lambda x: x.split('_'))
    data_tokenized['labels'] = data_tokenized.apply(
        lambda row: generate_labels(row['text'], row['keywords']), axis=1)

    texts = data_tokenized['text'].apply(lambda x: x.split()).tolist()

    labels = data_tokenized['labels'].tolist()
    # Encode labels
    label_encoder = LabelEncoder()
    flat_labels = [item for sublist in labels for item in sublist]
    label_encoder.fit(flat_labels)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    labels_encoded = [label_encoder.transform(label_seq) for label_seq in labels]

    tokenizer = Tokenizer(char_level=False)
    tokenizer.fit_on_texts(texts)
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    texts_encoded = tokenizer.texts_to_sequences(texts)

    max_len = max(max(len(seq) for seq in texts_encoded), max(len(seq) for seq in labels_encoded))

    texts_encoded_padded = pad_sequences(texts_encoded, maxlen=max_len, padding='post')
    labels_encoded_padded = pad_sequences(labels_encoded, maxlen=max_len, padding='post')

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(texts_encoded_padded, labels_encoded_padded, test_size=0.2, random_state=1)

    # Convert labels to categorical encoding for training
    y_train_categorical = np.array([np.eye(len(label_encoder.classes_))[seq] for seq in y_train])
    y_val_categorical = np.array([np.eye(len(label_encoder.classes_))[seq] for seq in y_val])

    print((X_train.shape, y_train_categorical.shape, X_val.shape, y_val_categorical.shape))
    #
    VOCAB_SIZE = len(tokenizer.word_index) + 1
    EMBEDDING_OUT_DIM = 128
    HIDDEN_UNITS = 64
    MAX_LEN = max_len
    NUM_CLASSES = len(label_encoder.classes_)
    config = {
        'VOCAB_SIZE': VOCAB_SIZE,
        'EMBEDDING_OUT_DIM': EMBEDDING_OUT_DIM,
        'HIDDEN_UNITS': HIDDEN_UNITS,
        'MAX_LEN': MAX_LEN,
        'NUM_CLASSES': NUM_CLASSES
    }

    with open('model_config.json', 'w') as f:
        json.dump(config, f)
    inputs = Input(shape=(MAX_LEN,), dtype='int32')
    output = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_OUT_DIM, trainable=True, mask_zero=True)(inputs)
    output = Bidirectional(LSTM(units=HIDDEN_UNITS, return_sequences=True))(output)

    crf = CRF(units=NUM_CLASSES, dtype='float32')
    output = crf(output)

    base_model = Model(inputs, output)


    model = ModelWithCRFLoss(base_model, sparse_target=True)

    model.compile(optimizer="adam")


    x = X_train
    y = y_train

    # Training the model
    history = model.fit(x=x, y=y, epochs=5, batch_size=32)  # Update epochs and batch_size as per your need

    base_model.save_weights('path_to_save_weights.h5')

