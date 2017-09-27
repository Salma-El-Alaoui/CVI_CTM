from src.variational_bayes_ctm.corpus import ToyDataset, NewsDataset, ApDataset
from src.variational_bayes_ctm.ctm_cvi import CTM_CVI
from src.variational_bayes_ctm.ctm import CTM
import numpy as np
import os

train_size = 0.7
data = NewsDataset(train_size=train_size)
output_directory = "../../results/20_news_groups"

number_topics = 20
em_iter = 50
e_iter = 30

cvi = True
step_size = 0.7

if cvi:
    filename = 'conv_cvi_S=%d.txt' % (step_size * 10)
    print("=====================CVI=====================")
    ctm = CTM_CVI(corpus=data.doc_set_train, vocab=data.vocabulary, number_of_topics=number_topics, em_max_iter=em_iter,
                  step_size=step_size, local_param_iter=e_iter)
else:
    filename = 'conv_vb'
    print("======================CTM-VB==================")
    ctm = CTM(corpus=data.doc_set_train, vocab=data.vocabulary, number_of_topics=number_topics, em_max_iter=em_iter,
              local_param_iter=e_iter)

lls_train, lls_test, times = ctm.fit_predict(data.doc_set_test)
print("lls_train: ", lls_train)
print("lls_test: ", lls_test)
print("times: ", times)
np.savetxt(os.path.join(output_directory, filename), (np.array(lls_test), np.array(times)))
