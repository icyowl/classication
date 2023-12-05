import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model, svm


with open('data/data_a.pickle', 'rb') as f:
    data = pickle.load(f)
    X, y = data

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# clf = svm.SVC(gamma=0.001)
# clf = MLPClassifier(max_iter=10000)
clf = linear_model.LogisticRegression(penalty='l2', C=10.0)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))

with open('data/clf_lr.pickle', 'wb') as f:
    pickle.dump(clf, f)