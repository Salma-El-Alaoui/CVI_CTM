from corpus import ToyDataset, NewsDataset
from ctm import CTM
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from sklearn import svm
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from plot_utils import plot_precision_recall
from sklearn.model_selection import cross_val_score
import os
def classification(number_topics, data=NewsDataset(), train_index=None, test_index=None, save=True, perform_class=True):
    output_directory = "../../results/20_news_groups/classification"
    print("Number Topics", number_topics)

    if train_index is None:
        doc_set_train = data.doc_set_train
        y_train_full = list(data.y_train)
        lambda_train_name = os.path.join(output_directory, 'lambda_train_' + str(number_topics) + '.txt')
        lambda_test_name = os.path.join(output_directory, 'lambda_test_' + str(number_topics) + '.txt')
        nus_train_name = os.path.join(output_directory, 'nus_train_' + str(number_topics) + '.txt')
        nus_test_name = os.path.join(output_directory, 'nus_test_' + str(number_topics) + '.txt')
    else:
        doc_set_train = [" ".join(d) for d in data.vectorizer.inverse_transform(data.X[train_index])]
        y_train_full = list(data.targets[train_index])

    if test_index is None:
        doc_set_test = data.doc_set_test
        y_test_full = list(data.y_test)
    else:
        doc_set_test = [" ".join(d) for d in data.vectorizer.inverse_transform(data.X[test_index])]
        y_test_full = list(data.targets[test_index])

    if save:
        ctm = CTM(corpus=doc_set_train, vocab=data.vocabulary, number_of_topics=number_topics)
        _, perplexity = ctm.fit()
        document_log_likelihood, perplexity, lambda_values, nu_square_values = ctm.predict(doc_set_test)
        if train_index is None:
            np.savetxt(lambda_train_name, ctm._lambda)
            np.savetxt(nus_train_name, ctm._nu_square)
            np.savetxt(lambda_test_name, lambda_values)
            np.savetxt(nus_test_name, nu_square_values)
            lambda_values_train = ctm._lambda
            nus_values_train = ctm._nu_square
    else:
        lambda_values_train = np.loadtxt(lambda_train_name)
        nus_values_train = np.loadtxt(nus_train_name)
        lambda_values = np.loadtxt(lambda_test_name)
        nu_square_values = np.loadtxt(nus_test_name)

    y_train = list()
    for i, doc in enumerate(data.doc_set_train):
        if doc != '':
            y_train.append(y_train_full[i])

    y_test = list()
    for i, doc in enumerate(data.doc_set_test):
        if doc != '':
            y_test.append(y_test_full[i])

    if perform_class:
        for C in range(1, 16):
            print("C", C)
            clf = svm.SVC(C=C, random_state=0)
            clf.fit(lambda_values_train, y_train)
            print("Accuracy", clf.score(lambda_values, y_test))

        # Y_train = label_binarize(y_train, classes=list(range(0, 20)))
        # Y_test = label_binarize(y_test, classes=list(range(0, 20)))
        # clf_p = OneVsRestClassifier(svm.SVC(C=C, random_state=0))
        # clf_p.fit(lambda_values_train, Y_train)
        # y_score = clf_p.decision_function(lambda_values)
        #
        # # For each class
        # precision = dict()
        # recall = dict()
        # average_precision = dict()
        # for i in range(20):
        #     precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
        #     average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
        #
        # # A "micro-average": quantifying score on all classes jointly
        # precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(), y_score.ravel())
        # average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")
        # print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))
        #
        # precision_name = os.path.join(output_directory, 'ctm_precision_'+str(number_topics)+'.npy')
        # recall_name = os.path.join(output_directory, 'ctm_recall_'+str(number_topics)+'.npy')
        # avg_name = os.path.join(output_directory, 'ctm_avg_precision_'+str(number_topics)+'.npy')
        #
        # np.save(precision_name, precision)
        # np.save(recall_name, recall)
        # np.save(avg_name, average_precision)
        #
        # prec_load = np.load(precision_name).item()
        # rec_load = np.load(recall_name).item()
        # avg_load = np.load(avg_name).item()

        # plot_precision_recall(prec_load, rec_load, avg_load)
        return lambda_values_train, lambda_values, y_train, y_test


def cross_validation(K):
    dataset = NewsDataset()
    accuracies_split = []
    for train_index, test_index in dataset.splits:
        print("================= new split =================")
        lambda_train, lambda_test, y_train, y_test = classification(number_topics=K, data=dataset,
                                                                    train_index=train_index, test_index=test_index,
                                                                    perform_class=False)
        accuracies = []
        for C in range(1, 16):
            print("C", C)
            clf = svm.SVC(C=C, random_state=0)
            clf.fit(lambda_train, y_train)
            accuracy = clf.score(lambda_test, y_test)
            print("Accuracy", accuracy)
            accuracies.append(accuracy)
        best = max(accuracies)
        accuracies_split.append(best)
        print("---------Best Accuracy---------", best)
    print("============== Cross validation accuracy ============", accuracies_split)
    arr_accuracy = np.array(accuracies_split)
    print("Mean", np.mean(arr_accuracy))
    print("STD", np.std(arr_accuracy))

if __name__ == "__main__":
    cross_validation(20)
