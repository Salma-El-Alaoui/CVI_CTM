import numpy as np
import string

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

        self.vocab = [string.ascii_lowercase[i] for i in range(vocab_size)]
        self.doc_set = [" ".join(d) for d in list_docs]

