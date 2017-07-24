# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from cvi import CVI

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
categories = dataset.target_names[:n_samples]
data_samples = dataset.data[:n_samples]

#%%
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
X = tf_vectorizer.fit_transform(data_samples)

#%%
lda = CVI(n_topics=n_topics, evaluate_every=5, learning_method='batch', max_iter=100, verbose=True, random_state=0)
lda.fit(X)

#%%
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

