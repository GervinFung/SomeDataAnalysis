import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def print_classifier_result(classifier_model_cv, x_train, x_test, y_train, y_test, y_predict, classifier_model_type):
    print(classifier_model_type + " Grid Search CV Best Score {}".format(classifier_model_cv.best_score_))
    print(classifier_model_type + " Grid Search CV Best Estimator {}".format(classifier_model_cv.best_estimator_))

    print(classifier_model_type + " Grid Search CV Training Score {}".
          format(classifier_model_cv.score(x_train, y_train)))
    print(classifier_model_type + " Grid Search CV Testing Score {}\n".
          format(classifier_model_cv.score(x_test, y_test)))
    print(classification_report(y_test, y_predict))


def confusion_matrix_graph(y_test, y_predict, classifier_model_cv):
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_test, y_predict, [0, 1])

    group_names = ['NEGATIVE(TRUE)', 'POSITIVE(FALSE)', 'NEGATIVE(FALSE)', 'POSITIVE(TRUE)']
    group_counts_dtc = ["{0:0.0f}".format(value) for value in confusion_matrix.flatten()]

    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts_dtc)]
    labels = np.asarray(labels).reshape(2, 2)

    sb.heatmap(confusion_matrix, annot=labels, fmt='', cmap='Reds')

    plt.title("CONFUSION MATRIX of " + classifier_model_cv)
    plt.xlabel("PREDICTED")
    plt.ylabel("ACTUAL")
    plt.show()


def compute_roc(y_test, y_predict):
    auc = roc_auc_score(y_test, y_predict)
    print('AUC: %.2f' % auc)
    return roc_curve(y_test, y_predict)


def roc_curve_graph(y_test, y_predict, classifier_model_type):
    fpr, tpr, thresholds = compute_roc(y_test, y_predict)
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve of ' + classifier_model_type)
    plt.legend()
    plt.show()


def logistic_regression(x_train, x_test, y_train, y_test, GridSearchCV, np):
    from sklearn.linear_model import LogisticRegression
    grid_parameters = {"C": np.arange(start=1, stop=51)}

    logistic_regression_cv = GridSearchCV(LogisticRegression(solver='lbfgs', max_iter=7600), grid_parameters, verbose=1, cv=10, n_jobs=-1)
    logistic_regression_cv.fit(x_train, y_train)

    y_predict = logistic_regression_cv.predict(x_test)
    classification_model = "Logistic Regression"

    print_classifier_result(logistic_regression_cv, x_train, x_test, y_train, y_test, y_predict, classification_model)
    confusion_matrix_graph(y_test, y_predict, classification_model)
    roc_curve_graph(y_test, y_predict, classification_model)

    return logistic_regression_cv.best_estimator_


def knn_classifier(x_train, x_test, y_train, y_test, GridSearchCV, np):
    from sklearn.neighbors import KNeighborsClassifier
    grid_parameters = {
        'n_neighbors': np.arange(start=1, stop=51),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    knn_classifier_cv = GridSearchCV(KNeighborsClassifier(), grid_parameters, verbose=1, cv=5, n_jobs=-1)
    knn_classifier_cv.fit(x_train, y_train)

    y_predict = knn_classifier_cv.predict(x_test)
    classification_model = "KNN Classifier"

    print_classifier_result(knn_classifier_cv, x_train, x_test, y_train, y_test, y_predict, classification_model)
    confusion_matrix_graph(y_test, y_predict, classification_model)
    roc_curve_graph(y_test, y_predict, classification_model)

    return knn_classifier_cv.best_estimator_


def gaussian_nb(x_train, x_test, y_train, y_test, GridSearchCV, np):
    from sklearn.naive_bayes import GaussianNB
    grid_parameters = {'var_smoothing': [0.00000001, 0.000000001, 0.0000000001]}

    gaussian_nb_cv = GridSearchCV(GaussianNB(), grid_parameters, verbose=1, cv=10, n_jobs=-1)
    gaussian_nb_cv.fit(x_train, y_train)

    y_predict = gaussian_nb_cv.predict(x_test)
    classification_model = "Gaussian Naive Bayes"

    print_classifier_result(gaussian_nb_cv, x_train, x_test, y_train, y_test, y_predict, classification_model)
    confusion_matrix_graph(y_test, y_predict, classification_model)
    roc_curve_graph(y_test, y_predict, classification_model)

    return gaussian_nb_cv.best_estimator_


def decision_tree_classifier(x_train, x_test, y_train, y_test, GridSearchCV, np):
    from sklearn.tree import DecisionTreeClassifier
    grid_parameters = {
        'criterion': ['gini', 'entropy'],
        'max_depth': np.arange(start=1, stop=10),
        'min_samples_split': np.arange(start=2, stop=11),
        'min_samples_leaf': np.arange(start=1, stop=5)
    }

    dct_cv = GridSearchCV(DecisionTreeClassifier(), grid_parameters, verbose=1, cv=5, n_jobs=-1)
    dct_cv.fit(x_train, y_train)

    y_predict = dct_cv.predict(x_test)
    classification_model = "Decision Tree"

    print_classifier_result(dct_cv, x_train, x_test, y_train, y_test, y_predict, classification_model)
    confusion_matrix_graph(y_test, y_predict, classification_model)
    roc_curve_graph(y_test, y_predict, classification_model)

    return dct_cv.best_estimator_


def random_forest_classifier(x_train, x_test, y_train, y_test, GridSearchCV, np):
    from sklearn.ensemble import RandomForestClassifier
    grid_parameters = {
        'bootstrap': [True, False],
        'max_depth': np.arange(start=10, stop=101, step=10),
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'n_estimators': np.arange(start=100, stop=501, step=100)
    }

    random_forest_cv = GridSearchCV(RandomForestClassifier(), grid_parameters, verbose=1, cv=5, n_jobs=-1)
    random_forest_cv.fit(x_train, y_train)

    y_predict = random_forest_cv.predict(x_test)
    classification_model = "Random Forest"

    print_classifier_result(random_forest_cv, x_train, x_test, y_train, y_test, y_predict, classification_model)
    confusion_matrix_graph(y_test, y_predict, classification_model)
    roc_curve_graph(y_test, y_predict, classification_model)

    return random_forest_cv.best_estimator_


def stacking_classifier(best_logistic_regression, best_knn_classifier, best_gaussian_nb, best_decision_tree_classifier,
                        best_random_forest_classifier, x_train, x_test, y_train, y_test):
    from sklearn.ensemble import StackingClassifier

    estimators = [
        # ('random_forest_cv', best_random_forest_classifier),
        ('knn_classifier_cv', best_knn_classifier),
        ('dct_cv', best_decision_tree_classifier),
        ('gaussian_nb_cv', best_gaussian_nb)
    ]

    final_stacking_classifier = StackingClassifier(estimators=estimators, shuffle=False, use_probas=True,
                                                   final_estimator=best_logistic_regression)

    final_stacking_classifier.fit(x_train, y_train)

    print("Stacking Classifier Training Score {}".format(final_stacking_classifier.score(x_train, y_train)))
    print("Stacking Classifier Testing Score {}\n".format(final_stacking_classifier.score(x_test, y_test)))

    y_predict = final_stacking_classifier.predict(x_test)
    classification_model = 'Stacking Classifier'

    confusion_matrix_graph(y_test, y_predict, classification_model)
    roc_curve_graph(y_test, y_predict, classification_model)


def all_classifier(x_train, x_test, y_train, y_test):
    from sklearn.model_selection import GridSearchCV

    best_logistic_regression = logistic_regression(x_train, x_test, y_train, y_test, GridSearchCV, np)
    best_knn_classifier = knn_classifier(x_train, x_test, y_train, y_test, GridSearchCV, np)
    best_gaussian_nb = gaussian_nb(x_train, x_test, y_train, y_test, GridSearchCV, np)
    best_decision_tree_classifier = decision_tree_classifier(x_train, x_test, y_train, y_test, GridSearchCV, np)
    # best_random_forest_classifier = random_forest_classifier(x_train, x_test, y_train, y_test, GridSearchCV, np)
    best_random_forest_classifier = 0

    stacking_classifier(best_logistic_regression, best_knn_classifier, best_gaussian_nb, best_decision_tree_classifier,
                        best_random_forest_classifier, x_train, x_test, y_train, y_test)
