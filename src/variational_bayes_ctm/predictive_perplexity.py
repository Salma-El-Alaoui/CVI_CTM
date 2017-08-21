from corpus import ToyDataset, NewsDataset
from ctm import CTM
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from plot_utils import plot_pp_train_per
import os
import gensim
import logging

if __name__ == "__main__":

    train_sizes = np.linspace(0.1, 0.9, num=9)
    #train_sizes = [0.7]
    output_directory = "../../results/20_news_groups"
    #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    perplexities_tr = list()
    perplexities_te = list()
    perplexities_te_lda = list()
    perplexities_tr_lda = list()
    for size in train_sizes:
        data = NewsDataset(train_size=size)
        number_topics = 20

        print("===============Observed words==============", size)

        print("-----------------LDA-----------------------")
        dictionary = gensim.corpora.Dictionary(data.doc_train_list)
        corpus = [dictionary.doc2bow(doc) for doc in data.doc_train_list]
        lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=number_topics,
                                              random_state=0, passes=20, decay=0, offset=0, chunksize=len(corpus))
        train_bound = lda.bound(corpus)
        pw_train_bound = lda.log_perplexity(corpus)
        print('final log-likelihood: %.4f, final log-perplexity: %.4f' % (train_bound, pw_train_bound))
        perplexities_tr_lda.append(pw_train_bound)
        test_chunk = [dictionary.doc2bow(doc) for doc in data.doc_test_list]
        gamma, _ = lda.inference(test_chunk)
        nb_words = sum(cnt for document in test_chunk for _, cnt in document)
        test_bound = lda.bound(test_chunk, gamma)
        pw_test_bound = lda.bound(test_chunk, gamma) / nb_words
        print('heldout log-likelihood: %.4f, heldout log-perplexity: %.4f' % (test_bound, pw_test_bound))
        perplexities_te_lda.append(pw_test_bound)

        print("-----------------CTM-----------------------")
        ctm = CTM(corpus=data.doc_set_train, vocab=data.vocabulary, number_of_topics=number_topics)
        _, perplexity = ctm.fit()
        _, heldout_perplexity, _, _ = ctm.predict(test_corpus=data.doc_set_test)
        perplexities_tr.append(perplexity)
        perplexities_te.append(heldout_perplexity)

    print("Train CTM: ", perplexities_tr)
    print("Train LDA: ", perplexities_tr_lda)
    print("Test CTM: ", perplexities_te)
    print("Test LDA: ", perplexities_te_lda)

    np.savetxt(os.path.join(output_directory, 'log_perplexities_ctm_lda.txt'), (np.asarray(perplexities_tr), np.asarray(perplexities_tr_lda),
               np.asarray(perplexities_te), np.asarray(perplexities_te_lda)))

    plot_pp_train_per(x=train_sizes, inspectors=[perplexities_te, perplexities_te_lda], models=["CTM", "LDA"])
