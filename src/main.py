# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from cvi import CVI
import numpy as np

#%%
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
#%%
n_samples = 2000
n_features = 1000
n_topics = 20
n_top_words = 10

dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'))
categories = dataset.target[:n_samples]
categories_names = dataset.target_names[:n_samples]
data_samples = dataset.data[:n_samples]

#%%
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
X = tf_vectorizer.fit_transform(data_samples)

#%%
#ctm = CVI(n_topics=n_topics, evaluate_every=5, learning_method='batch', max_iter=100, verbose=True, random_state=0)
#ctm.fit(X)
#%%
#print("=================CTM================")
tf_feature_names = tf_vectorizer.get_feature_names()
#print_top_words(ctm, tf_feature_names, n_top_words)

#%%
lda = LatentDirichletAllocation(n_topics=n_topics, learning_method='batch', max_iter=100, verbose=False, random_state=0)
lda.fit(X)
print(lda.perplexity(X))
#%%
print("=================LDA================")
print_top_words(lda, tf_feature_names, n_top_words)

#%%
cnts = X.todense()
topics = np.zeros((n_topics, n_features))
for topic in np.unique(categories):
    for document, topic_d in enumerate(categories):
        if topic == topic_d:
            topics[topic, :] += cnts[document, :].A1
#%%

for topic_idx, topic in enumerate(topics): 
    print(categories_names[topic_idx], ":")
    print(" ".join([tf_feature_names[i] for i in topic.argsort()[:-n_top_words -1:-1]]))
print()
