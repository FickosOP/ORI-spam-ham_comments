import os
import string
# import nltk
import math
import time
from inflector import Inflector
from tabulate import tabulate
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
# from nltk.stem import WordNetLemmatizer

DIRECTORY_PATH = 'Comments'

STOP_WORDS = ['is', 'was', 'the', 'on', 'a', 'i', 'an', 'and', 'of', 'for', 'as', 'us']

UNCOUNTABLE_WORDS = ['this', 'was', 'his', 'has', 'plus', 'police', 'bless', 'dress', 'pass', 'jesus', 'less', 'cross',
                     'as', 'mess', 'cross', 'mess', 'miss']


def parse_word(word):
    lowered = word.lower()
    no_punc = lowered.translate(str.maketrans('', '', string.punctuation))
    if no_punc in stopwords.words('english'):
        return ""

    if no_punc.startswith("http"):
        return "link"
    no_punc.replace("'", "")
    inf = Inflector()
    sing = no_punc

    if sing not in UNCOUNTABLE_WORDS:
        sing = inf.singularize(no_punc)
    # if sing != no_punc:
    #     print("JEDNINA: " + sing)
    if len(sing) < 2:
        return ""
    # STEMATIZACIJA PO POTREBI
    return sing


def create_corpus():
    COMMENT_NUM = 0
    root = os.path.dirname(__file__)
    directory = os.path.join(root, DIRECTORY_PATH)
    corpus = []
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            file_path = f"{directory}\{file}"
            with open(file_path, 'r', encoding="utf-8-sig") as f:
                for line in f.readlines():
                    corpus.append(line)
                    COMMENT_NUM += 1
    # corpus = corpus[:1376] 70% trening skup
    return corpus


def preprocess_text():
    processed_data = []
    for line in create_corpus():
        processed_comment = []
        for word in line.strip().split(',')[3].split(" "):
            parsed_word = parse_word(word)
            if parsed_word == "":
                continue
            processed_comment.append(parsed_word)
        processed_data.append(processed_comment)
    return processed_data


def create_word_freq_dict(data):
    wordfreq = {}
    for parsed_comment in data:
        for parsed_word in parsed_comment:
            if parsed_word not in wordfreq.keys():
                wordfreq[parsed_word] = 1
            else:
                wordfreq[parsed_word] += 1

    new_dict = wordfreq.copy()

    for key in wordfreq.keys():
        if wordfreq[key] < 3:
            new_dict.pop(key)
    print(len(new_dict))
    return new_dict


def create_bag_of_words(data):
    word_freq_dict = create_word_freq_dict(data)
    sentence_vectors = []  # bag of words
    for sentence_tokens in data:
        sent_vec = []
        for token in word_freq_dict:
            if token in sentence_tokens:
                sent_vec.append(1)
            else:
                sent_vec.append(0)
        sentence_vectors.append(sent_vec)
    return sentence_vectors


def create_tf_idf(data):
    tf_idf = {}  # for each word calculate tf-idf as key(word)-val(tf_idf)
    cnt_comment = 0
    for comment in data:
        cnt_comment += 1
        for word in comment:
            tf = calculate_tf(word, comment)
            idf = calculate_idf(word, data)
            tf_idf[cnt_comment, word] = tf * idf
    return tf_idf


def calculate_tf(word, comment):
    cnt = 0
    for w in comment:
        if w == word:
            cnt += 1
    return cnt/len(comment)


def calculate_idf(word, comments):
    cnt = 0
    for com in comments:
        if word in com:
            cnt += 1
    return math.log(len(comments) / (cnt + 1))


def show_bof(bof, data):
    print(tabulate(bof[:10], headers=create_word_freq_dict(data).keys()))


def show_idf(idf, data):
    first_ten_vals = []

    for key in idf.keys():
        first_ten_vals.append(idf[key])
    # print(first_ten_vals)
    print(tabulate([first_ten_vals], headers=idf.keys()))


if __name__ == '__main__':
    print("Loading bag of words...")

    # start = time.time()
    #
    # processed_text = preprocess_text()
    # bag_of_words_model = create_bag_of_words(processed_text)
    #
    # end = time.time()
    # print(end - start)
    # show_bof(bag_of_words_model, processed_text)

    print("Loading TF-IDF...")
    # start = time.time()
    # tf_idf_model = create_tf_idf(processed_text)
    # end = time.time()
    # print(end - start)
    #
    # show_idf(tf_idf_model, processed_text)
    # -----------------------------------------------------------------------

    df = pd.concat(
        [pd.read_csv('D:\\Machine Learning_Algoritms\\Youtube-Spam-Check\\Youtube01-Psy.csv', encoding='latin1'),
         pd.read_csv('D:\\Machine Learning_Algoritms\\Youtube-Spam-Check\\Youtube02-KatyPerry.csv', encoding='latin1'),
         pd.read_csv('D:\\Machine Learning_Algoritms\\Youtube-Spam-Check\\Youtube03-LMFAO.csv', encoding='latin1'),
         pd.read_csv('D:\\Machine Learning_Algoritms\\Youtube-Spam-Check\\Youtube04-Eminem.csv', encoding='latin1'),
         pd.read_csv('D:\\Machine Learning_Algoritms\\Youtube-Spam-Check\\Youtube05-Shakira.csv', encoding='latin1')])
    # print(f"\t\tDATAFRAME\n{df}")
    X = df.drop('CLASS', axis=1)
    Y = df['CLASS']
    # print(f"\t\tX:\n{X}")
    #     # print(f"\t\tY:\n{Y}")

    vectorizer = CountVectorizer()

    dv = vectorizer.fit_transform(df['CONTENT'])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    dtrain_att = vectorizer.fit_transform(X_train['CONTENT'])  # fit bag-of-words model to training set
    print(f"METODI FIT_TRANSFORM SMO PROSLEDILI: {X_train['CONTENT']}")
    print(f"PRE TRENING SKUPA: {len(X_train)}")
    print("TRENING SKUP:\n")
    train_cnt = 0
    for row in dtrain_att:
        train_cnt += 1
    print(f"DIM: {train_cnt}")
    print(dtrain_att)
    # <- ovo sam napravi (dtrain_att i d_test_att kljuc je par
    # (redni broj komentara, kolona komentara(koja je rec po redu))
    #  vrednost je koliko se puta pojavljuje)) --
    test_att = vectorizer.transform(X_test['CONTENT'])

    RFC = RandomForestClassifier(n_estimators=10, random_state=0)  # kreiraj klasifikator

    RFC.fit(dtrain_att, y_train)   # fit trening skup
    #  umesto dtrain att prosledi custom svoj

    y_pred = RFC.predict(test_att)   # prediktuj vrednost nad test skupom
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))  # izracunaj tacnost
    print("F1:", metrics.f1_score(y_test, y_pred))  # izracunaj tacnost

    tf_vectorizer = TfidfVectorizer()
    dv_tf_idf = tf_vectorizer.fit_transform(X_train['CONTENT'])
    test_att = tf_vectorizer.transform(X_test['CONTENT'])

    RFC.fit(dv_tf_idf, y_train)
    y_pred = RFC.predict(test_att)  # prediktuj vrednost nad test skupom
    print("Accuracy TF-IDF:", metrics.accuracy_score(y_test, y_pred))  # izracunaj tacnost
    print("F1:", metrics.f1_score(y_test, y_pred))  # izracunaj tacnost
    # RFC.fit(bag_of_words_model, None)
    #
    # RFC.score(bag_of_words_model, None)
#  calculate idf u metodu create tfidf
