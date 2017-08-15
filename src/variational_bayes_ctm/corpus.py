import numpy as np
import string
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import re
import os
from sklearn.model_selection import train_test_split

class ToyDataset:

    def __init__(self, nb_topics=3, nb_documents=100, vocab_size=5, document_size=50, concentration=1/3, ctm=True):
        # Constants
        self.K = nb_topics
        self.D = nb_documents
        self.V = vocab_size
        self.N = document_size
        self.ctm = ctm
        self.concentration = concentration

        # hyper-parameters:
        self.mu = np.zeros(self.K)
        self.sigma = 0.1 * np.identity(self.K)
        self.gamma = concentration * np.ones(self.V)

        # Draw topic
        self.beta = np.random.dirichlet(alpha=self.gamma, size=self.K)

        # Draw topic proportion for each document
        if ctm:
            self.eta = np.random.multivariate_normal(mean=self.mu, cov=self.sigma, size=self.D)
            self.theta = np.exp(self.eta) / np.sum(np.exp(self.eta), axis=1)[:, None]
        else:
            self.theta_prior = concentration * np.ones(self.K)
            self.theta = np.random.dirichlet(alpha=self.theta_prior, size=self.D)

        # Draw topic assignment for each document and each word
        self.Z = np.asarray([np.random.multinomial(n=1, pvals=self.theta[d, :], size=self.N) for d in range(self.D)])

        # Draw Words:
        self.W = np.asarray(
            [[np.random.multinomial(n=1, pvals=self.beta[np.where(self.Z[d, n, :] == 1)[0].squeeze()])
                for n in range(self.N)]
                for d in range(self.D)])

        list_docs = [[string.ascii_lowercase[(np.where(self.W[d, n, :] == 1)[0].squeeze())]
                        for n in range(self.N)]
                        for d in range(self.D)]

        self.vocabulary = [string.ascii_lowercase[i] for i in range(vocab_size)]
        self.doc_set = [" ".join(d) for d in list_docs]


class NewsDataset:

    def __init__(self, n_samples=None, shuffle=True, random_state=0, train_size=0.8):
        input_directory = "../../data/20_news_groups"
        vocabulary_path = os.path.join(input_directory, 'vocabulary.txt')
        input_voc_stream = open(vocabulary_path, 'r')
        vocab = []
        for line in input_voc_stream:
            vocab.append(line.strip().lower().split()[0])
        self.vocabulary = list(set(vocab))
        self.shuffle = True
        self.random_state = random_state
        self.train_size = train_size

        dataset = fetch_20newsgroups(shuffle=shuffle, random_state=self.random_state, remove=('headers', 'footers', 'quotes'))
        if n_samples is None:
            self.n_samples = len(dataset.target)
        else:
            self.n_samples = n_samples

        self.categories = dataset.target[:n_samples]
        self.categories_names = dataset.target_names[:n_samples]
        self.raw_samples = dataset.data[:n_samples]

        vectorizer = CountVectorizer(max_df=0.95, min_df=2, vocabulary=self.vocabulary, preprocessor=self.preprocessor) #stop_words='english'
        self.X = vectorizer.fit_transform(self.raw_samples)
        self.X_train, self.X_test = train_test_split(self.X, train_size=self.train_size, random_state=self.random_state)
        self.doc_set_train = [" ".join(d) for d in vectorizer.inverse_transform(self.X_train)]
        self.doc_set_test = [" ".join(d) for d in vectorizer.inverse_transform(self.X_test)]

    @staticmethod
    def preprocessor(doc):
        doc = doc.lower()
        doc = re.sub(r'-', ' ', doc)
        doc = re.sub(r'[^a-z ]', '', doc)
        doc = re.sub(r' +', ' ', doc)
        return doc