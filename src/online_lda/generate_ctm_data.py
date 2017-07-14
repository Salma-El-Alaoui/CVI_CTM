import numpy as np
import os
import sys

# Add parent directory to python path
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from online_lda.onlineldavb import OnlineLDA,


class ToyDataset:

    def __init__(self, nb_topics=3, nb_documents=100, vocab_size=5, document_size=50, concentration=0.5, ctm=True):
        # Constants
        self.K = nb_topics
        self.D = nb_documents
        self.V = vocab_size
        self.N = document_size
        self.ctm = ctm

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

        # Variable to use external Online LDA
        list_docs = [[str(np.where(self.W[d, n, :] == 1)[0].squeeze())
                        for n in range(self.N)]
                        for d in range(self.D)]
        self.vocab = [str(i) for i in range(vocab_size)]

        self.doc_set = [" ".join(d) for d in list_docs]

if __name__ == "__main__":
    data = ToyDataset(ctm=False)
    # alpha is the hyperparameter for prior on weight vectors theta
    # eta is the hyperparameter for prior on topics beta
    # doing batch LDA here
    lda = OnlineLDA(vocab=data.vocab, K=data.K, D=data.D, alpha=data.theta_prior, eta=data.gamma, tau0=0, kappa=0)
    iterations = 1000
    old_perplexity = 1.0 * sys.maxsize
    delta_perplexity = 1.0 * sys.maxsize
    for i in range(iterations):
        gamma, bound = lda.update_lambda_docs(data.doc_set)
        word_ids, word_cts = lda.parse_doc_list(data.doc_set, lda._vocab)
        # estimate perpexity with the current batch
        perplexity = bound * len(data.doc_set) / (data.D * sum(map(sum, word_cts)))
        delta_perplexity = abs(old_perplexity - perplexity) / perplexity
        print('iteration = %d: perplexity estimate = %f (%.2f%%)' % (i, np.exp(-perplexity), delta_perplexity * 100))
        old_perplexity = perplexity



