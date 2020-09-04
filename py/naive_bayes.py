import time
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from keras.utils import to_categorical
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import f_classif, SelectKBest
import nltk
from textblob import Word
nltk.download("stopwords")
nltk.download('wordnet')


def fold_stat(pred_train, pred_test, y_train, y_test):
    tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train[:, 1], pred_train).ravel()
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test[:, 1], pred_test).ravel()

    return tn_train, fp_train, fn_train, tp_train, tn_test, fp_test, fn_test, tp_test

def extract_data():
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


def vertorize_and_select(x_train, x_test,y_train, k):
    print("TF-IDF Vectorize.........")
    cv = TfidfVectorizer(sublinear_tf=True, max_df=0.5, ngram_range=(1, 3))
    # cv = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
    # cv = CountVectorizer(binary=True, max_df=0.5, ngram_range=(1, 3))
    # cv = CountVectorizer(binary=True, max_df=0.5)


    x_train_tf = cv.fit_transform(x_train)
    x_test_tf = cv.transform(x_test)


    print("Selecting K Best Features..........")
    selector = SelectKBest(f_classif, k=k)
    selector.fit(x_train_tf, y_train)
    x_train_tf = selector.transform(x_train_tf)
    x_test_tf = selector.transform(x_test_tf)

    # transfer the data into array
    x_train_tf = x_train_tf.toarray()
    x_test_tf = x_test_tf.toarray()
    return x_train_tf, x_test_tf


def one_hot_encode(y_train, y_test):
    return to_categorical(y_train, 2), to_categorical(y_test, 2)



df_x, df_y = extract_data()
# x_train_tf, x_test_tf, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# y_train = to_categorical(y_train, 2)
# y_test = to_categorical(y_test, 2)
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
for train, test in kf.split(df_x, df_y):
    count += 1
    print("=======Processing Fold {}=======".format(count))
    x_train = df_x[train]
    x_test = df_x[test]
    y_train = np.array(df_y[train], dtype=int)
    y_test = np.array(df_y[test], dtype=int)

    x_train_tf, x_test_tf = vertorize_and_select(x_train, x_test,y_train, 5000)

    model = MultinomialNB()
    start = time.process_time()
    model.fit(x_train_tf, y_train)

    total_time_cost += time.process_time() - start

    pred_train = model.predict(x_train_tf)
    pred_test = model.predict(x_test_tf)

    y_train_one_hot, y_test_one_hot = one_hot_encode(y_train, y_test)
    tn_train, fp_train, fn_train, tp_train, tn_test, fp_test, fn_test, tp_test = fold_stat(pred_train, pred_test, y_train_one_hot, y_test_one_hot)

    total_tn_test += tn_test
    total_fp_test += fp_test
    total_fn_test += fn_test
    total_tp_test += tp_test

    total_tn_train += tn_train
    total_fp_train += fp_train
    total_fn_train += fn_train
    total_tp_train += tp_train


avg_time = total_time_cost / n_splits

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
# print("----------")
# print("Time Cost: {}".format(avg_time))
