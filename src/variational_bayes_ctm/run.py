from corpus import ToyDataset, NewsDataset
from ctm import CTM
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from plot_utils import plot_pp_train_per
import os

if __name__ == "__main__":

    train_sizes = np.linspace(0.1, 0.9, num=9)
    output_directory = "../../results/20_news_groups"

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
        lda = LatentDirichletAllocation(n_topics=20, learning_method='batch', max_iter=100, verbose=True,
                                        random_state=0)
        lda.fit(data.X_train)
        perplexity_lda = lda.perplexity(data.X_train)
        heldout_perplexity_lda = lda.perplexity(data.X_test)
        print("perplexity ", perplexity_lda)
        print("heldout perplexity", heldout_perplexity_lda)
        perplexities_tr_lda.append(perplexity_lda)
        perplexities_te_lda.append(heldout_perplexity_lda)

    print("Train CTM: ", perplexities_tr)
    print("Train LDA: ", perplexities_tr_lda)
    print("Test CTM: ", perplexities_te)
    print("Test LDA: ", perplexities_te_lda)

    np.savetxt(os.path.join(output_directory, 'perplexities.txt'), (np.asarray(perplexities_tr), np.asarray(perplexities_tr_lda),
               np.asarray(perplexities_te), np.asarray(perplexities_te_lda)))

    plot_pp_train_per(x=train_sizes, inspectors=[perplexities_te], models=[CTM])
