from corpus import ToyDataset, NewsDataset, ApDataset
from ctm_cvi import CTM_CVI
from ctm import CTM
import numpy as np

train_size = 0.7
data = NewsDataset(train_size=train_size)
number_topics = 20
em_iter = 50

print("=====================CVI=====================")
ctm = CTM_CVI(corpus=data.doc_set_train, vocab=data.vocabulary, number_of_topics=number_topics, em_max_iter=em_iter,
              step_size=0.7)
lls_train, lls_test, times = ctm.fit_predict(data.doc_set_test)
print("lls_train: ", lls_train)
print("lls_test: ", lls_test)
print("times: ", times)