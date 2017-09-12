from corpus import ToyDataset, NewsDataset, ApDataset
from ctm_cvi import CTM_CVI
from ctm import CTM
import numpy as np

train_size = 0.7
data = NewsDataset(train_size=train_size)
number_topics = 20

print("=====================CVI=====================")
ctm = CTM_CVI(corpus=data.doc_set_train, vocab=data.vocabulary, number_of_topics=number_topics)
_, perplexity = ctm.fit()
_, heldout_perplexity, _, _ = ctm.predict(test_corpus=data.doc_set_test)
print()
print("=====================VB=====================")
ctm = CTM(corpus=data.doc_set_train, vocab=data.vocabulary, number_of_topics=number_topics)
_, perplexity = ctm.fit()
_, heldout_perplexity, _, _ = ctm.predict(test_corpus=data.doc_set_test)
