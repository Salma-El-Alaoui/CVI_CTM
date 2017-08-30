from corpus import ToyDataset, NewsDataset, ApDataset
from ctm import CTM
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import os
import gensim
import logging


def perplexity_ctm(train_sizes, dataset, number_topics, save=False, output_directory=None):
    perplexities_tr = list()
    perplexities_te = list()
    print("-----------------CTM-----------------------")
    print("Number of topics ", number_topics)
    for size in train_sizes:
        data = dataset(train_size=size)
        print("===============Observed words==============", size)
        ctm = CTM(corpus=data.doc_set_train, vocab=data.vocabulary, number_of_topics=number_topics)
        _, perplexity = ctm.fit()
        _, heldout_perplexity, _, _ = ctm.predict(test_corpus=data.doc_set_test)
        perplexities_tr.append(perplexity)
        perplexities_te.append(heldout_perplexity)

    if save:
        print("Train CTM: ", perplexities_tr)
        print("Test CTM: ", perplexities_te)
        np.savetxt(os.path.join(output_directory, str(number_topics)+'_log_perplexities_ctm_observed.txt'),
                   (train_sizes, np.asarray(perplexities_tr), np.asarray(perplexities_te)))

    return perplexities_tr, perplexities_te


def perplexity_topics(topic_numbers, train_size, dataset, output_directory):
    print("-----------------CTM-----------------------")
    print("Train size ", train_size)
    for t in topic_numbers:
        print("===============Number of topics==============", t)
        perplexities_tr, perplexities_te = perplexity_ctm(train_sizes=[train_size], dataset=dataset, number_topics=t)
        print("Train CTM: ", perplexities_tr)
        print("Test CTM: ", perplexities_te)
        np.savetxt(os.path.join(output_directory, str(train_size) + '_log_perplexities_ctm_topics.txt'),
                   (topic_numbers, np.asarray(perplexities_tr), np.asarray(perplexities_te)))


if __name__ == "__main__":
    out = "../../results/20_news_groups"
    d = NewsDataset
    k = 10
    perplexity_ctm(train_sizes=np.linspace(0.1, 0.9, num=9), dataset=d, number_topics=k, save=True, output_directory=out)



