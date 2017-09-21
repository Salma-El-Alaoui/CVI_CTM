from corpus import ToyDataset, NewsDataset, ApDataset
from ctm_cvi_stochastic import CTM_CVI_S
from ctm import CTM
import numpy as np

train_size = 0.7
data = NewsDataset(train_size=train_size)
number_topics = 20
em_iter = 20
e_iter = 20
step_size = 0.7

print("=====================CVI=====================")
ctm = CTM_CVI_S(corpus=data.doc_set_train, vocab=data.vocabulary, number_of_topics=number_topics, em_max_iter=em_iter,
              step_size=step_size, local_param_iter=e_iter)
ctm.fit()
ctm.predict(test_corpus=data.doc_set_test)