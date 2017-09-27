from src.variational_bayes_ctm.corpus import ToyDataset, NewsDataset, ApDataset
from src.variational_bayes_ctm.ctm import CTM
import numpy as np

train_size = 0.99
data = NewsDataset(train_size=train_size)
number_topics = 20
ctm = CTM(corpus=data.doc_set_train, vocab=data.vocabulary, number_of_topics=number_topics)
_, perplexity = ctm.fit()
_, heldout_perplexity, _, _ = ctm.predict(test_corpus=data.doc_set_test)
topics_path = "../../results/20_news_groups/%d_beta_topics.txt" % number_topics
ctm.export_beta(exp_beta_path=topics_path, top_display=20)
