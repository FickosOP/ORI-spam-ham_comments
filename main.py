import os
import string
# import nltk
import math
import time
from inflector import Inflector
from tabulate import tabulate
from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

DIRECTORY_PATH = 'Comments'

STOP_WORDS = ['is', 'was', 'the', 'on', 'a', 'i', 'an', 'and', 'of', 'for', 'as', 'us']

UNCOUNTABLE_WORDS = ['this', 'was', 'his', 'has', 'plus', 'police', 'bless', 'dress', 'pass', 'jesus', 'less', 'cross',
                     'as', 'mess', 'cross', 'mess', 'miss', 'piss']


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
    root = os.path.dirname(__file__)
    directory = os.path.join(root, DIRECTORY_PATH)
    corpus = []
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            file_path = f"{directory}\{file}"
            with open(file_path, 'r', encoding="utf-8-sig") as f:
                for line in f.readlines():
                    corpus.append(line)
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

    start = time.time()

    processed_text = preprocess_text()
    bag_of_words_model = create_bag_of_words(processed_text)

    end = time.time()
    print(end - start)
    # show_bof(bag_of_words_model, processed_text)

    print("Loading TF-IDF...")
    start = time.time()
    tf_idf_model = create_tf_idf(processed_text)
    end = time.time()
    print(end - start)

    print(tf_idf_model)
    # show_idf(tf_idf_model, processed_text)

#  calculate idf u metodu create tfidf
