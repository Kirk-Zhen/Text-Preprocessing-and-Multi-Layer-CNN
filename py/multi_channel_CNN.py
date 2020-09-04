import time
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, LSTM, Embedding, Input, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical
from keras.layers.merge import concatenate
from keras.models import Model
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import nltk
from textblob import Word
nltk.download("stopwords")
nltk.download('wordnet')


def fold_stat(pred_train, pred_test, y_train, y_test):
    tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train[:, 1], pred_train).ravel()
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test[:, 1], pred_test).ravel()

    return tn_train, fp_train, fn_train, tp_train, tn_test, fp_test, fn_test, tp_test

def extract_data():
    df = pd.read_csv("/content/drive/My Drive/COMP4107/data/IMDB Dataset.csv")
    print("Doing NLTK")
    df['review'] = df['review'].apply(lambda x: " ".join(x.lower() for x in x.split()))

     # remove digits, punctuation and words of less than 3 characters
    df['review'] = df['review'].str.replace('[^\w\s]', '')
    df['review'] = df['review'].str.replace('\d+', '')  # for digits
    df['review'] = df['review'].str.replace(r'(\b\w{1,2}\b)', '')  # for words
    df['review'] = df['review'].str.replace(r'\s+', ' ')
    # stop = stopwords.words('english')
    # df['review'] = df['review'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    df['review'] = df['review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    df.loc[df["sentiment"] == "positive", "sentiment"] = 1
    df.loc[df["sentiment"] == "negative", "sentiment"] = 0
    #
    # # store the NLTK processed data into .csv file
    # # df.to_csv("data/IMDB_modified.csv", index=False)

    # df = pd.read_csv("/content/drive/My Drive/COMP4107/data/IMDB_modified.csv")

    # df = pd.read_csv("/content/drive/My Drive/COMP4107/data/IMDB Dataset.csv")
    # df.loc[df["sentiment"] == "positive", "sentiment"] = 1
    # df.loc[df["sentiment"] == "negative", "sentiment"] = 0

    df_x = df["review"]
    df_y = df["sentiment"]
    df_x = np.array(df_x)
    df_y = np.array(df_y).astype(float)
    # x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=24)

    return df_x, df_y


def one_hot_encode(y_train, y_test):
    return to_categorical(y_train, 2), to_categorical(y_test, 2)

def multichannel_model(length,vocab_size):
    #Channel 1
    inputs1 = Input(shape=(length,), name='unigram_input')
    embedding1 = Embedding(vocab_size,64, name="unigram_embed")(inputs1)
    conv1 = Conv1D(filters=32,kernel_size=1,activation='relu',name = "unigram_conv1d")(embedding1)
    drop1 = Dropout(0.5, name='unigram_dropout')(conv1)
    pool1 = MaxPooling1D(pool_size=1,name='unigram_pooling')(drop1)
    flat1 = Flatten(name='unigram_flatten')(pool1)
    # flat1 = Flatten(name='unigram_flatten')(drop1)



    inputs2 = Input(shape=(length,), name='bigram_input')
    embedding2 = Embedding(vocab_size,64, name="bigram_embed")(inputs2)
    conv2 = Conv1D(filters=32,kernel_size=2,activation='relu',name = "bigram_conv1d")(embedding2)
    drop2 = Dropout(0.5, name='bigram_dropout')(conv2)
    pool2 = MaxPooling1D(pool_size=1,name='bigram_pooling')(drop2)
    flat2 = Flatten(name='bigram_flatten')(pool2)
    # flat2 = Flatten(name='bigram_flatten')(drop2)

    #Channel 3
    inputs3 = Input(shape=(length,), name='trigram_input')
    embedding3 = Embedding(vocab_size,64, name="trigram_embed")(inputs3)
    conv3 = Conv1D(filters=32,kernel_size=3,activation='relu',name = "trigram_conv1d")(embedding3)
    drop3 = Dropout(0.5, name='trigram_dropout')(conv3)
    pool3 = MaxPooling1D(pool_size=1,name='trigram_pooling')(drop3)
    flat3 = Flatten(name='trigram_flatten')(pool3)
    # flat3 = Flatten(name='trigram_flatten')(drop3)

    #merging channels
    merged = concatenate([flat1,flat2,flat3], name="feature_combine")

    #interpretation
    dense1 = Dense(64,activation='relu', name="64-Neurons")(merged)
    outputs = Dense(2,activation='softmax', name="Output_Layer")(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

    #Compiling
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    #Summarize
    model.summary()
    #plot_model(model,show_shapes=True,to_file = 'multichannel.png')
    return model



df = pd.read_csv("data/IMDB Dataset.csv")
print("Doing NLTK")
df['review'] = df['review'].apply(lambda x: " ".join(x.lower() for x in x.split()))
 # remove digits, punctuation and words of less than 3 characters
df['review'] = df['review'].str.replace('[^\w\s]', '')
df['review'] = df['review'].str.replace('\d+', '')  # for digits
df['review'] = df['review'].str.replace(r'(\b\w{1,2}\b)', '')  # for words
df['review'] = df['review'].str.replace(r'\s+', ' ')
# stop = stopwords.words('english')
# df['review'] = df['review'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df['review'] = df['review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

df.loc[df["sentiment"] == "positive", "sentiment"] = 1
df.loc[df["sentiment"] == "negative", "sentiment"] = 0

reviews = df["review"]
sentiment = df["sentiment"]

max_features = 5000
# tokenizer = Tokenizer(num_words=max_features, filters='')
tokenizer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\d+')
tokenizer.fit_on_texts(reviews)
df_x = tokenizer.texts_to_sequences(reviews)
df_x = pad_sequences(df_x)

# x_train_tf, x_test_tf, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# y_train = to_categorical(y_train, 2)
# y_test = to_categorical(y_test, 2)
embed_size = 64
length = df_x.shape[1]
n_splits = 5
kf = KFold(n_splits=5)
sum_acc_test = 0
sum_acc_train = 0
total_time_cost = 0

total_tn_test = 0
total_fp_test = 0
total_fn_test = 0
total_tp_test = 0

total_tn_train = 0
total_fp_train = 0
total_fn_train = 0
total_tp_train = 0
print("=======Evaluating {} Fold Experiment=======".format(n_splits))
count = 0
for train, test in kf.split(df_x, sentiment):
    count += 1
    x_train_tf = df_x[train]
    x_test_tf = df_x[test]
    y_train = np.array(sentiment[train], dtype=int)
    y_test = np.array(sentiment[test], dtype=int)
    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)



    model = multichannel_model(length, 5000)

    start = time.process_time()

    model.fit([x_train_tf, x_train_tf, x_train_tf], y_train,
              epochs=3,
              batch_size=64,
              validation_data=([x_test_tf, x_test_tf, x_test_tf], y_test))

    total_time_cost += time.process_time() - start


    pred_train = model.predict([x_train_tf, x_train_tf, x_train_tf])
    pred_test = model.predict([x_test_tf, x_test_tf, x_test_tf])

    pred_train = np.argmax(pred_train, axis=1)
    pred_test = np.argmax(pred_test, axis=1)
    tn_train, fp_train, fn_train, tp_train, tn_test, fp_test, fn_test, tp_test = fold_stat(pred_train, pred_test, y_train, y_test)

    total_tn_test += tn_test
    total_fp_test += fp_test
    total_fn_test += fn_test
    total_tp_test += tp_test

    total_tn_train += tn_train
    total_fp_train += fp_train
    total_fn_train += fn_train
    total_tp_train += tp_train

avg_tn_train = total_tn_train / n_splits
avg_fp_train = total_fp_train / n_splits
avg_fn_train = total_fn_train / n_splits
avg_tp_train = total_tp_train / n_splits

avg_tn_test = total_tn_test / n_splits
avg_fp_test = total_fp_test / n_splits
avg_fn_test = total_fn_test / n_splits
avg_tp_test = total_tp_test / n_splits

print("----------")
print("Train TN: {}".format(avg_tn_train))
print("Train FP: {}".format(avg_fp_train))
print("Train FN: {}".format(avg_fn_train))
print("Train TP: {}".format(avg_tp_train))
# Precision, Recall, f1 score for Test
precision_train = avg_tp_train / (avg_tp_train + avg_fp_train)
recall_train = avg_tp_train / (avg_tp_train + avg_fn_train)
f1_score_train = 2 * ((precision_train * recall_train) / (precision_train + recall_train))
print("Train Precision: {}".format(precision_train))
print("Train Recall: {}".format(recall_train))
print("Train F1 Score: {}".format(f1_score_train))

# confusion matrix for Test
print("----------")
print("Test TN: {}".format(avg_tn_test))
print("Test FP: {}".format(avg_fp_test))
print("Test FN: {}".format(avg_fn_test))
print("Test TP: {}".format(avg_tp_test))
# Precision, Recall, f1 score for Test
precision_test = avg_tp_test / (avg_tp_test + avg_fp_test)
recall_test = avg_tp_test / (avg_tp_test + avg_fn_test)
f1_score_test = 2 * ((precision_test * recall_test) / (precision_test + recall_test))
print("Test Precision: {}".format(precision_test))
print("Test Recall: {}".format(recall_test))
print("Test F1 Score: {}".format(f1_score_test))
