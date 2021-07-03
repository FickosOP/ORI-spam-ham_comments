import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pandas as pd

DIRECTORY_PATH = 'Comments'

if __name__ == '__main__':
    root = os.path.dirname(__file__)
    directory = os.path.join(root, DIRECTORY_PATH)
    corpus = []
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            file_path = f"{directory}\{file}"
            corpus.append(pd.read_csv(file_path))
    df = pd.concat(corpus)

    X = df.drop('CLASS', axis=1)
    Y = df['CLASS']

    bag_of_words_vector = CountVectorizer()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    train_set_bof = bag_of_words_vector.fit_transform(X_train['CONTENT'])
    test_set_bof = bag_of_words_vector.transform(X_test['CONTENT'])

    RFC = RandomForestClassifier(n_estimators=80, random_state=0)

    d_shuffle = df.sample(frac=1)
    d_content = d_shuffle['CONTENT']
    d_label = d_shuffle['CLASS']

    pl_1 = make_pipeline(CountVectorizer(), TfidfTransformer(norm=None), RandomForestClassifier())
    pl_1.fit(d_content[:1500], d_label[:1500])

    pl_2 = make_pipeline(CountVectorizer(), RandomForestClassifier())
    pl_2.fit(d_content[:1500], d_label[:1500])

    pl_3 = make_pipeline(CountVectorizer(), MultinomialNB())
    pl_3.fit(d_content[:1500], d_label[:1500])

    pl_4 = make_pipeline(TfidfVectorizer(), MultinomialNB())
    pl_4.fit(d_content[:1500], d_label[:1500])

    RFC.fit(train_set_bof, Y_train)

    bof_prediction = RFC.predict(test_set_bof)   # predict values for test set
    print("\n\t\t-- BAG OF WORDS + RANDOM FOREST --")
    print("Accuracy:", metrics.accuracy_score(Y_test, bof_prediction))  # calculate accuracy
    print("\t  F1:", metrics.f1_score(Y_test, bof_prediction))  # calculate F1
    conf_mat_1 = confusion_matrix(bof_prediction, Y_test)
    print(f"Confusion matrix: \n{conf_mat_1}")

    tf_vector = TfidfVectorizer()
    train_tf_idf = tf_vector.fit_transform(X_train['CONTENT'])
    test_set_tf_idf = tf_vector.transform(X_test['CONTENT'])

    RFC.fit(train_tf_idf, Y_train)

    tf_idf_prediction = RFC.predict(test_set_tf_idf)
    print("\n\t\t-- TF-IDF + RANDOM FOREST --")
    print("Accuracy:", metrics.accuracy_score(Y_test, tf_idf_prediction))
    print("\t  F1:", metrics.f1_score(Y_test, tf_idf_prediction))
    conf_mat_2 = confusion_matrix(tf_idf_prediction, Y_test)
    print(f"Confusion matrix: \n{conf_mat_2}")
    NB = MultinomialNB()

    NB.fit(train_set_bof, Y_train)

    nb_prediction = NB.predict(test_set_bof)
    print("\n\t\t-- BAG OF WORDS + NAIVE BAYES --")
    print(f"Accuracy: {metrics.accuracy_score(Y_test, nb_prediction)}")
    print("\t  F1:", metrics.f1_score(Y_test, nb_prediction))
    conf_mat_3 = confusion_matrix(nb_prediction, Y_test)
    print(f"Confusion matrix: \n{conf_mat_3}")

    NB.fit(train_tf_idf, Y_train)
    nb_prediction = NB.predict(test_set_tf_idf)
    print("\n\t\t-- TF-IDF + NAIVE BAYES --")
    print(f"Accuracy: {metrics.accuracy_score(Y_test, nb_prediction)}")
    print("\t  F1:", metrics.f1_score(Y_test, nb_prediction))
    conf_mat_4 = confusion_matrix(nb_prediction, Y_test)
    print(f"Confusion matrix: \n{conf_mat_4}")

    while True:
        input_comment = input("Enter your comment to see if it is SPAM or HAM: ")
        result_pl1 = pl_1.predict([input_comment])
        print(f"\nTF-IDF + Random forest prediction: \t{'Spam' if result_pl1 == 1 else 'Ham'}")

        result_pl2 = pl_2.predict([input_comment])
        print(f"\nBag of words + Random forest prediction: \t{'Spam' if result_pl2 == 1 else 'Ham'}")

        result_pl3 = pl_3.predict([input_comment])
        print(f"\nBag of words + Naive Bayes prediction: \t{'Spam' if result_pl3 == 1 else 'Ham'}")

        result_pl4 = pl_4.predict([input_comment])
        print(f"\nTF-IDF + Naive Bayes prediction: \t{'Spam' if result_pl3 == 1 else 'Ham'}")
