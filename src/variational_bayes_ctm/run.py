from corpus import ToyDataset, NewsDataset
from ctm import CTM
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

if __name__ == "__main__":

    train_sizes = np.linspace(0.1, 0.9, num=9)
    perplexities_tr = list()
    perplexities_te = list()
    perplexities_te_lda = list()
    perplexities_tr_lda = list()
    for size in train_sizes:
        data = NewsDataset(train_size=size)
        print("===============Observed words==============", size)
        ctm = CTM(corpus=data.doc_set_train, vocab=data.vocabulary, number_of_topics=20)
        print("-----------------CTM-----------------------")
        _, perplexity = ctm.fit()
        _, heldout_perplexity, _, _ = ctm.predict(test_corpus=data.doc_set_test)
        perplexities_tr.append(perplexity)
        perplexities_te.append(heldout_perplexity)
        print("-----------------LDA-----------------------")
        lda = LatentDirichletAllocation(n_topics=20, learning_method='online', max_iter=100, verbose=True,
                                        random_state=0)
        lda.fit(data.X_train)
        perplexity_lda = lda.score(data.X_train)
        heldout_perplexity_lda = lda.perplexity(data.X_test)
        print("perplexity ", perplexity_lda)
        print("heldout perplexity", heldout_perplexity_lda)
        perplexities_tr_lda.append(perplexity_lda)
        perplexities_te_lda.append(heldout_perplexity_lda)
        #np.savetxt('2perplexity_' + str(size) + '.txt', (np.asarray(perplexities_tr), np.asarray(perplexities_te)))
    print("TRAIN CTM", perplexities_tr)
    print("TRAIN LDA", perplexities_tr_lda)
    print("TEST CTM", perplexities_te)
    print("TEST LDA", perplexities_te_lda)
