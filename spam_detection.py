import os
import string
# import nltk
import time
from NaiveBayesClassifier import *
from scipy.sparse import csr_matrix
from inflector import Inflector
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd

DIRECTORY_PATH = 'Comments'


def load_dataframe():
    """
    Function that opens all csv files from DIRECTORY_PATH location and adds them to corpus
    :return: Data-frame of processed comments, count of unique words, list of unique words
    """
    print("Loading data-frame...")
    root = os.path.dirname(__file__)
    directory = os.path.join(root, DIRECTORY_PATH)
    corpus = []
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            file_path = f"{directory}\{file}"
            corpus.append(pd.read_csv(file_path))
    df = pd.concat(corpus)
    df, word_cnt, words = pre_process_text(df)
    return df, word_cnt, words


def pre_process_text(df):
    """
    Function that will replace current comment with new processed comment
    :param df: data-frame with all attributes, processing will be performed only on CONTENT
    :return: data-frame with processed text
    """
    print("Processing text...")
    start = time.time()
    word_cnt = 0
    processed = []  # dictionary
    for row in df['CONTENT']:
        new_content = ""  # new comment
        for word in row.split(" "):
            parsed = parse_word(word)
            if parsed != "":
                new_content += parsed + " "
                if parsed not in processed:
                    word_cnt += 1
                    processed.append(parsed)
        df = df.replace([row], new_content)
    end = time.time()
    print(f"Text processing finished in {end - start} s.")
    return df, word_cnt, processed


def parse_word(word):
    """
    Parsing single word of comment. Actions on word:
    1. Lower
    2. Drop punctuation
    3. Ignore if stop word
    4. Singularity
    5. Drop words that contain 1 or 2 characters
    :param word: Non-processed word
    :return: Processed word
    """
    lowered = word.lower()
    no_punctuation = lowered.translate(str.maketrans('', '', string.punctuation))
    if no_punctuation in stopwords.words('english'):
        return ""

    if no_punctuation.startswith("http"):
        return "link"
    no_punctuation.replace("'", "")
    inf = Inflector()
    sing = no_punctuation

    uncountable_words = ['this', 'his', 'has', 'plus', 'police',
                         'bless', 'dress', 'pass', 'jesus', 'less',
                         'cross', 'mess', 'cross', 'mess', 'miss']

    if sing not in uncountable_words:
        sing = inf.singularize(no_punctuation)

    if len(sing) < 2:
        return ""
    # STEMMING DIDN'T IMPROVE ACCURACY IT JUST SLOWED DOWN TEXT PROCESSING
    return sing


def create_bag_of_words(comments, wc, word_list):
    """
    Creating bag of words representation as csr matrix
    key: (comment index, word index) value: number of appearances in i. comment
    :param comments: Processed comments from data-frame
    :param wc: Word count for whole data-frame
    :param word_list: List of all unique words in data-frame
    :return: Bag of words model (csr matrix)
    """
    i = 0
    data = []
    row = []
    col = []
    for comment in comments:
        com_tokens = comment.split(' ')
        for word in com_tokens[:-1]:
            if word != '':
                count = com_tokens.count(word)
                row.append(i)
                # int instead of word(columns are represented as words)
                j = word_list.index(word)
                col.append(j)
                data.append(count)
        i += 1
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    csr = csr_matrix((data, (row, col)), shape=(len(comments), wc))
    return csr


def create_tf_idf(comments, wc, word_list):
    """
    Creating tf-idf representation as csr matrix
    key: (comment index, word index) value: tf * idf
    tf (TERM FREQUENCY) = appearances in single comment / word count of that comment
    idf (INVERSE DOCUMENT FREQUENCY) = log()
    :param comments: Processed comments from data-frame
    :param wc: Word count for whole data-frame
    :param word_list: List of all unique words in data-frame
    :return: TF-IDF model (csr matrix)
    """
    i = 0
    data = []
    row = []
    col = []
    for comment in comments:
        com_tokens = comment.split(' ')
        for word in com_tokens[:-1]:
            if word != '':
                count = com_tokens.count(word)
                row.append(i)
                # int instead of word(columns are represented as words)
                j = word_list.index(word)
                col.append(j)
                tf = count / len(com_tokens[:-1])
                idf = calculate_idf(word, comments)
                tf_idf = abs(tf * idf)
                data.append(tf_idf)
        i += 1
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    csr = csr_matrix((data, (row, col)), shape=(len(comments), wc))
    return csr


def calculate_idf(word, comments):
    """
    Calculates idf value for one word
    :param word: String
    :param comments: List of all comments
    :return: Returns log(N / (df+1) )
             N -> number of comments
             df -> how many comments contain word 'word'
    """
    cnt = 0
    for com in comments:
        if word in com:
            cnt += 1
    return np.log(len(comments) / (cnt + 1))


def predict_input(input_text, all_ds_words):
    """
    Classifies input comment as SPAM or HAM using TF-IDF + RFC
    :param input_text: comment entered by user
    :param all_ds_words: Isolated words from all comments
    :return:
    """
    parsed_comment = ""
    for word in input_text.split(' '):
        parsed = parse_word(word)
        if parsed != "" and parsed in all_ds_words:
            parsed_comment += parsed + " "
    tf_idf_input = create_tf_idf([parsed_comment], len(all_ds_words), all_ds_words)

    RFC.fit(d_train_tf_idf, Y_train)
    res = RFC.predict(tf_idf_input)[0]
    if res == 1:
        return True
    return False


def create_confusion_matrix(results, reals):
    """
    Creates dictionary that represents confusion matrix and prints out results
    Counts TP, TN, FP, FN
    TP - number of correctly predicted spam comments
    TN - number of correctly predicted ham comments
    FP - number of ham comments predicted as spam
    FN - number of spam comments predicted as ham
    :param results: List of predicted values
    :param reals: List of actual values
    :return: Confusion matrix as dictionary
    """
    conf_mat = {}
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    if len(results) != len(reals):
        return conf_mat

    i = 0
    for real_v in reals:
        res_v = results[i]
        if real_v == 0:
            if res_v == 0:
                tn += 1  # Real class is HAM (0), model predicted HAM (0)
            elif res_v == 1:
                fp += 1  # Real class is HAM (0), model predicted SPAM (1)
        elif real_v == 1:
            if res_v == 0:
                fn += 1  # Real class is SPAM (1), model predicted HAM (0)
            elif res_v == 1:
                tp += 1  # Real class is SPAM (1), model predicted SPAM (1)
        i += 1
    conf_mat["TP"] = tp
    conf_mat["TN"] = tn
    conf_mat["FP"] = fp
    conf_mat["FN"] = fn
    print("{:40}{:10}{:10}".format("Actual/Predicted", "Spam", "Ham"))
    print("{:40}{:4}{:8}".format("Spam", tp, fn))
    print("{:40}{:4}{:8}".format("Ham", fp, tn))

    return conf_mat


if __name__ == '__main__':
    data_frame, word_count, all_words = load_dataframe()
    X = data_frame.drop('CLASS', axis=1)  # everything but tag
    Y = data_frame['CLASS']  # tags

    #  creating train and test data set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    #  passing only comments
    d_train_bof = create_bag_of_words(X_train['CONTENT'], word_count, all_words)
    d_test_bof = create_bag_of_words(X_test['CONTENT'], word_count, all_words)

    d_train_tf_idf = create_tf_idf(X_train['CONTENT'], word_count, all_words)
    d_test_tf_idf = create_tf_idf(X_test['CONTENT'], word_count, all_words)

    RFC = RandomForestClassifier(n_estimators=80, random_state=0)

    # -- BAG OF WORDS -- #

    RFC.fit(d_train_bof, Y_train)

    y_prediction_bof = RFC.predict(d_test_bof)

    print(f"Accuracy for Bag Of Words model: {metrics.accuracy_score(Y_test, y_prediction_bof)}\n")

    confusion_matrix = create_confusion_matrix(y_prediction_bof, Y_test)

    scores = cross_val_score(RFC, d_train_bof, Y_train, cv=5)
    print(f"\nCross-validation with 5 splits: {scores.mean()}")

    # -- TF-IDF -- #
    RFC.fit(d_train_tf_idf, Y_train)

    y_prediction_tf_idf = RFC.predict(d_test_tf_idf)
    print(f"\n\nAccuracy for TF-IDF model: {metrics.accuracy_score(Y_test, y_prediction_tf_idf)}\n")

    confusion_matrix_tf = create_confusion_matrix(y_prediction_tf_idf, Y_test)

    scores_tf = cross_val_score(RFC, d_train_tf_idf, Y_train, cv=5)
    print(f"\nCross-validation with 5 splits: {scores_tf.mean()}")

    # -- NAIVE BAYES -- #
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.fit(X_train['CONTENT'], Y_train)

    nb_prediction = nb_classifier.test(X_test['CONTENT'])
    print(f"\n\nAccuracy with Naive Bayes: {metrics.accuracy_score(Y_test, nb_prediction)}\n")

    confusion_matrix_nb = create_confusion_matrix(nb_prediction, Y_test)

    while True:
        print("Enter a comment to see if it is spam or ham: ")
        input_comment = input()
        if input_comment == 'X':
            exit(0)
        if predict_input(input_comment, all_words):
            print("SPAM")
        else:
            print("HAM")
