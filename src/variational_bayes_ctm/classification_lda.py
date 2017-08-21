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
import os
import gensim
import logging

if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    output_directory = "../../results/20_news_groups"
    data = NewsDataset(train_size=0.9)
    number_topics = 20

    dictionary = gensim.corpora.Dictionary(data.doc_train_list)
    corpus = [dictionary.doc2bow(doc) for doc in data.doc_train_list]
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=number_topics,
                                          random_state=0, passes=15, decay=0, offset=0, chunksize=len(corpus))
    train_bound = lda.bound(corpus)
    pw_train_bound = lda.log_perplexity(corpus)
    print('final log-likelihood: %.4f, final log-perplexity: %.4f' % (train_bound, pw_train_bound))
    test_chunk = [dictionary.doc2bow(doc) for doc in data.doc_test_list]
    gamma, _ = lda.inference(test_chunk)
    nb_words = sum(cnt for document in test_chunk for _, cnt in document)
    test_bound = lda.bound(test_chunk, gamma)
    pw_test_bound = lda.bound(test_chunk, gamma) / nb_words
    print('heldout log-likelihood: %.4f, heldout log-perplexity: %.4f' % (test_bound, pw_test_bound))

    document_topic_train = lda.get_document_topics(bow=corpus, minimum_probability=0)
    train_data = list()
    test_data = list()
    for doc in range(len(document_topic_train)):
        train_data.append([topic_dis for _, topic_dis in document_topic_train[doc]])
    document_topic_test = lda.get_document_topics(bow=test_chunk, minimum_probability=0)
    for doc in range(len(document_topic_test)):
        test_data.append([topic_dis for _, topic_dis in document_topic_test[doc]])

    C = 50

    clf = svm.SVC(C=C, random_state=0)
    clf.fit(train_data, data.y_train)
    print("Accuracy", clf.score(test_data, data.y_test))

    Y_train = label_binarize(data.y_train, classes=list(range(0, 20)))
    Y_test = label_binarize(data.y_test, classes=list(range(0, 20)))
    clf_p = OneVsRestClassifier(svm.SVC(C=C, random_state=0))
    clf_p.fit(train_data, Y_train)
    y_score = clf_p.decision_function(test_data)

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(20):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))

    precision_name = os.path.join(output_directory, 'ctm_precision_' + str(number_topics) + '.npy')
    recall_name = os.path.join(output_directory, 'ctm_recall_' + str(number_topics) + '.npy')
    avg_name = os.path.join(output_directory, 'ctm_avg_precision_' + str(number_topics) + '.npy')

    np.save(precision_name, precision)
    np.save(recall_name, recall)
    np.save(avg_name, average_precision)

    prec_load = np.load(precision_name).item()
    rec_load = np.load(recall_name).item()
    avg_load = np.load(avg_name).item()

    plot_precision_recall(prec_load, rec_load, avg_load)
