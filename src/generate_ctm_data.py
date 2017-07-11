import numpy as np

class ToyDataset:

    def __init__(self, nb_topics=3, nb_documents=100, vocab_size=5, document_size=50, concentration=0.1):

        # Constants
        self.K = nb_topics
        self.D = nb_documents
        self.V = vocab_size
        self.N = document_size
        self.gamma = concentration

        # hyperparameters:
        self.mu = np.zeros(self.K)
        self.sigma = 0.1 * np.identity(self.K)
        self.gamma = self.gamma * np.ones(self.V)

        # Draw topic
        self.beta = np.random.dirichlet(alpha=self.gamma, size=self.K)

        # Draw topic proportion for each document
        self.eta = np.random.multivariate_normal(self.mu, self.sigma, self.D)
        theta = np.exp(self.eta) / np.sum(np.exp(self.eta), axis=1)[:, None]

        # Draw topic assignment for each document and each word
        self.Z = [np.random.multinomial(n=1, pvals=theta[d, :], size=self.N) for d in range(self.D)]
        print(self.Z[0])

        # Draw Words:
        #self.W = [np.random.multinomial(n=1, pvals=self.beta[d, ], size=self.) for d in range(self.D)]



data = ToyDataset()