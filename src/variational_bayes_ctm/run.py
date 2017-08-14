from corpus import ToyDataset, NewsDataset
from ctm import CTM

if __name__ == "__main__":
    data = NewsDataset()
    ctm = CTM(corpus=data.doc_set, vocab=data.vocabulary, number_of_topics=10)
    ctm.fit()


    #from sklearn.decomposition import LatentDirichletAllocation
    #print(data.X.sum())
    #lda = LatentDirichletAllocation(n_topics=10, learning_method='online', max_iter=1000, verbose=True, random_state=0,
    #                               evaluate_every=10)
    #lda.fit(data.X)
    #print(lda.perplexity(data.X))