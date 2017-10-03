from src.variational_bayes_ctm.corpus import ToyDataset, NewsDataset, ApDataset, DeNewsDataset
from src.variational_bayes_ctm.ctm_cvi import CTM_CVI
from src.variational_bayes_ctm.cvi_stochastic_expectation import CTM_CVI_SE
import numpy as np
import os

train_size = 0.7
data = DeNewsDataset(train_size=train_size)

number_topics = 10
em_iter = 100
e_iter = 30

cvi = True
step_size = 0.7

if cvi:
    filename = 'conv_cvi_S=%d.txt' % (step_size * 10)
    print("=====================CVI=====================")
    ctm = CTM_CVI_SE(corpus=data.doc_set_train, vocab=data.vocabulary, number_of_topics=number_topics,
                     em_max_iter=em_iter, step_size=step_size, local_param_iter=e_iter)

lls_train, lls_test, times = ctm.fit_predict(data.doc_set_test)


