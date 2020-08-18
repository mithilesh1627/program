import sklearn
from sklearn.datasets import fetch_openml

mnist = fetch_openml(name='mnist_784' , version=1)
print(mnist.keys())

import matplotlib.pyplot as plt
import matplotlib as mlp
X,y = mnist['data'],mnist['target']
import matplotlib as mpl
import matplotlib.pyplot as plt
some_digit = X[0]
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image, cmap = 'binary')
plt.axis('off')
plt.show()
print(y[0])

import numpy as np

y = y.astype(np.uint8)
print(y)
X_train , X_test , y_train , y_test = X[:60000],X[60000:],y[:60000],y[60000:]

y_train_5 = (y_train ==5)
y_test_5 = (y_test ==5)
from  sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train_5)
sgd_clf.predict([some_digit])
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
 clone_clf = clone(sgd_clf)
 X_train_folds = X_train[train_index]
 y_train_folds = y_train_5[train_index]
 X_test_fold = X_train[test_index]
 y_test_fold = y_train_5[test_index]
 clone_clf.fit(X_train_folds, y_train_folds)
 y_pred = clone_clf.predict(X_test_fold)
 n_correct = sum(y_pred == y_test_fold)
 print(n_correct / len(y_pred))

from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf , X_train , y_train_5 ,cv=3 )
#
# from sklearn.base import BaseEstimator
# class Neverclassifier5(BaseEstimator):
#     def fit(self,X,y=None):
#         return self
#     def predict(self,X):
#         return np.zeros((len(X),1),dtype= bool)
#
# never_5_clf = Neverclassifier5()
# cross_val_score(never_5_clf,X_train,y_train_5,cv=3)

from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf,X_train,y_train_5,cv=3)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train_5,y_train_pred))

y_prefect_predictions =y_train_5
print(confusion_matrix(y_train_5,y_prefect_predictions))

from sklearn.metrics import  precision_score ,recall_score
print(precision_score(y_train_5,y_train_pred))

print(recall_score(y_train_5,y_train_pred))

y_score = sgd_clf.decision_function([some_digit])
y_score
threshold=0
y_some_digit_pred = (y_score > threshold)
print(y_some_digit_pred)

threshold=8000
y_some_digit_pred = (y_score > threshold)
print(y_some_digit_pred)

y_score = cross_val_predict(sgd_clf , X_train , y_train_5 , cv=3 ,
                            method='decision_function')
from sklearn.metrics import precision_recall_curve
precisions,recalls,thresholds = precision_recall_curve(y_train_5,y_score)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
 plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
 plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

from sklearn.metrics import roc_curve
fpr , tpr , thresholds = roc_curve(y_train_5,y_score)


def plot_roc_curve(fpr, tpr, label=None):
 plt.plot(fpr, tpr, linewidth=2, label=label)
 plt.grid(True)
 plt.xlabel('False positive rate')
 plt.ylabel('True positive rate')
 plt.plot([0, 1], [0, 1], 'k--')
 plt.xlim(0,1)
 plt.ylim(0,1)
 plt.title('ROC Curve')

plot_roc_curve(fpr, tpr)
plt.show()


plot_roc_curve(fpr, tpr)
plt.show()

from sklearn.metrics import roc_auc_score
print('roc_aur_score :  ' , roc_auc_score(y_train_5, y_score))

print("now for randomforestclassifier ")
from sklearn.ensemble import RandomForestClassifier
forest_clf =RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf,X_train,y_train_5 , cv=3,
                                    method="predict_proba")

y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)


plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()

print('roc_auc_score : ' , roc_auc_score(y_train_5,y_scores_forest))



from sklearn.svm import SVC
svm_clf = SVC()
svm_clf.fit(X_train, y_train) # y_train, not y_train_5
svm_clf.predict([some_digit])


some_digit_scores = svm_clf.decision_function([some_digit])

print(some_digit_scores)

import numpy as np
np.argmax(some_digit_scores)
svm_clf.classes_
svm_clf.classes_[5]

from sklearn.multiclass import OneVsRestClassifier
ovr_clf = OneVsRestClassifier(SVC())
ovr_clf.fit(X_train, y_train)


ovr_clf.predict([some_digit])
print(len(ovr_clf.estimators_))


sgd_clf.fit(X_train, y_train)

sgd_clf.predict([some_digit])

sgd_clf.decision_function([some_digit])

cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")


y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
print(conf_mx)


plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()

cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
plt.show()

# multilabel classification system
from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")



noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)


print("-----------------------End of project  `-_-`----------------------")