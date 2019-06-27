import csv

from nltk import TweetTokenizer
from nltk.corpus import stopwords
from sklearn import model_selection, metrics
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import KFold


def load(csv_file):
    polaridade = list()
    opinioes = list()
    with open(csv_file, 'r', encoding='utf-8') as csv_file:
        try:
            reader = csv.reader(csv_file, delimiter='|')
            for row in reader:

                polaridade.append(int(str(row[0]).replace('\ufeff', '')))
                texto = row[2].lower()
                texto = TweetTokenizer().tokenize(texto)
                """for t in texto:
                    if 'https' in t:
                        texto.remove(t)
                    if '#' in t:
                        texto.remove(t)
                    if '@' in t:
                        texto.remove(t)
"""
                opinioes.append(texto)
        except IOError:
            pass
    return opinioes, polaridade


def pre():
    csv_file = 'corpus-politica.csv'
    x, y = load(csv_file)
    vectorizer = TfidfVectorizer()
    #vectorizer = TfidfVectorizer(ngram_range=(2, 2), analyzer='word')
    pt_br_stop_words = set(stopwords.words('portuguese'))

    for i in range(len(x)):
       #x[i] = [w for w in x[i] if not w in pt_br_stop_words]
        x[i] = ' '.join(x[i])
    x = vectorizer.fit_transform(x)

    return x, y


def pro():
    x, y = pre()
    kfold = KFold(n_splits=10)

    clfs = [
        ('svm', SVC()),
        ('mnb', MultinomialNB()),
        ('rl', LogisticRegression()),
        ('pa', PassiveAggressiveClassifier())
    ]

    voting = VotingClassifier(estimators=clfs)
    result = model_selection.cross_val_predict(voting, x, y, cv=kfold)
    print("\n Voting Result")
    print("Acuracia: %f" % metrics.accuracy_score(y, result))
    print("Precisaão: %f" % metrics.precision_score(y, result,  average='weighted'))
    print("F-score: %f" % metrics.f1_score(y, result,  average='weighted'))
    print("Recall: %f" % metrics.recall_score(y, result,  average='weighted'))
    print(metrics.confusion_matrix(y, result, labels=(1, 2, 3)))

    bagging = BaggingClassifier(n_estimators=10)
    result = model_selection.cross_val_predict(bagging, x, y, cv=kfold)
    print("\n Bagging Result")
    print("Acuracia: %f" % metrics.accuracy_score(y, result))
    print("Precisaão: %f" % metrics.precision_score(y, result, average='weighted'))
    print("F-score: %f" % metrics.f1_score(y, result, average='weighted'))
    print("Recall: %f" % metrics.recall_score(y, result, average='weighted'))
    print(metrics.confusion_matrix(y, result, labels=(1, 2, 3)))

    boosting = AdaBoostClassifier(n_estimators=10)
    result = model_selection.cross_val_predict(boosting, x, y, cv=kfold)
    print("\n Boosting Result")
    print("Acuracia: %f" % metrics.accuracy_score(y, result))
    print("Precisaão: %f" % metrics.precision_score(y, result, average='weighted'))
    print("F-score: %f" % metrics.f1_score(y, result, average='weighted'))
    print("Recall: %f" % metrics.recall_score(y, result, average='weighted'))
    print(metrics.confusion_matrix(y, result, labels=(1, 2, 3)))


if __name__ == '__main__':
    pro()

