from corpus import ToyDataset, NewsDataset, ApDataset
from ctm_cvi import CTM_CVI
from ctm import CTM
import numpy as np

train_size = 0.7
data = NewsDataset(train_size=train_size)
number_topics = 20
em_iter = 50
e_iter = 20
cvi = True
step_size = 0.7

if cvi:
    print("=====================CVI=====================")
    ctm = CTM_CVI(corpus=data.doc_set_train, vocab=data.vocabulary, number_of_topics=number_topics, em_max_iter=em_iter,
                  step_size=step_size, local_param_iter=e_iter)
else:
    print("======================CTM-VB==================")
    ctm = CTM(corpus=data.doc_set_train, vocab=data.vocabulary, number_of_topics=number_topics, em_max_iter=em_iter,
              local_param_iter=e_iter)

lls_train, lls_test, times = ctm.fit_predict(data.doc_set_test)
print("lls_train: ", lls_train)
print("lls_test: ", lls_test)
print("times: ", times)