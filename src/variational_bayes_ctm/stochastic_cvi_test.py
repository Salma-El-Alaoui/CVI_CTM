from src.variational_bayes_ctm.corpus import ToyDataset, NewsDataset, ApDataset
from src.variational_bayes_ctm.ctm_cvi_stochastic import CTM_CVI_S
import numpy as np
import os

train_size = 0.7
data = NewsDataset(train_size=train_size)

number_topics = 20
em_iter = 50
e_iter = 30

step_size = 0.7
batch_size = 300
learning_offset = 64
learning_decay = 0.6
evaluate_every = 10
output_directory = "../../results/20_news_groups"
filename = 'conv_stoch_cvi_B=%d_OF=%d_D=%d_S=%d.txt' % (batch_size, learning_offset, learning_decay*10, step_size*10)

print("=====================STOCHASTIC CVI=====================")
ctm = CTM_CVI_S(corpus=data.doc_set_train, vocab=data.vocabulary, number_of_topics=number_topics, em_max_iter=em_iter,
                step_size=step_size, local_param_iter=e_iter, batch_size=batch_size, learning_offset=learning_offset,
                learning_decay=learning_decay, evaluate_every=evaluate_every)
lls_train, lls_test = ctm.fit_predict(data.doc_set_test)
print("lls_test: ", lls_test)
np.savetxt(os.path.join(output_directory, filename), np.array(lls_test))
