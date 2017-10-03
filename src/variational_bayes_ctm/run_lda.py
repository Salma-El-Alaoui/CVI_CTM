"""
Variational Bayesian Inference for Latent Dirichlet Allocation
This code was modified from the code originally written by Matthew Hoffman.
Implements Variational Bayes for LDA as described in (Blei et al. 2003)
"""

import sys, re, time, string
import numpy as n
from scipy.special import gammaln, psi
from src.variational_bayes_ctm.corpus import ToyDataset, NewsDataset, ApDataset, DeNewsDataset, NipsDataset
from sklearn import svm

meanchangethresh = 0.001
n.random.seed(100000001)


def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return (psi(alpha) - psi(n.sum(alpha)))
    return (psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])


def parse_doc_list(docs, vocab):
    """
    Parse a document into a list of word ids and a list of counts,
    or parse a set of documents into two lists of lists of word ids
    and counts.

    Arguments:
    docs:  List of D documents. Each document must be represented as
           a single string. (Word order is unimportant.) Any
           words not in the vocabulary will be ignored.
    vocab: Dictionary mapping from words to integer ids.

    Returns a pair of lists of lists.

    The first, wordids, says what vocabulary tokens are present in
    each document. wordids[i][j] gives the jth unique token present in
    document i. (Don't count on these tokens being in any particular
    order.)

    The second, wordcts, says how many times each vocabulary token
    present. wordcts[i][j] is the number of times that the token given
    by wordids[i][j] appears in document i.
    """
    if (type(docs).__name__ == 'str'):
        temp = list()
        temp.append(docs)
        docs = temp

    D = len(docs)

    wordids = list()
    wordcts = list()
    for d in range(0, D):
        docs[d] = docs[d].lower()
        docs[d] = re.sub(r'-', ' ', docs[d])
        docs[d] = re.sub(r'[^a-z ]', '', docs[d])
        docs[d] = re.sub(r' +', ' ', docs[d])
        words = str.split(docs[d])
        ddict = dict()
        for word in words:
            if (word in vocab):
                wordtoken = vocab[word]
                if (not wordtoken in ddict):
                    ddict[wordtoken] = 0
                ddict[wordtoken] += 1
        wordids.append(ddict.keys())
        wordcts.append(ddict.values())

    return ((wordids, wordcts))


class OnlineLDA:
    """
    Implements online VB for LDA as described in (Hoffman et al. 2010).
    """

    def __init__(self, vocab, K, alpha, eta, tau0, kappa):
        """
        Arguments:
        K: Number of topics
        vocab: A set of words to recognize. When analyzing documents, any word
           not in this set will be ignored.
        D: Total number of documents in the population. For a fixed corpus,
           this is the size of the corpus. In the truly online setting, this
           can be an estimate of the maximum number of documents that
           could ever be seen.
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        tau0: A (positive) learning parameter that downweights early iterations
        kappa: Learning rate: exponential decay rate---should be between
             (0.5, 1.0] to guarantee asymptotic convergence.

        Note that if you pass the same set of D documents in every time and
        set kappa=0 this class can also be used to do batch VB.
        """
        self._vocab = dict()
        for word in vocab:
            word = word.lower()
            if not word.isdigit():
                word = re.sub(r'[^a-z]', '', word)
            self._vocab[word] = len(self._vocab)

        self._K = K
        self._W = len(self._vocab)
        self._alpha = alpha
        self._eta = eta
        self._tau0 = tau0 + 1
        self._kappa = kappa
        self._updatect = 0

        # Initialize the variational distribution q(beta|lambda)
        self._lambda = 1 * n.random.gamma(100., 1. / 100., (self._K, self._W))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)

    def do_e_step(self, wordids, wordcts, iter=50):
        batchD = len(wordids)

        # Initialize the variational distribution q(theta|gamma) for
        # the mini-batch
        gamma = 1 * n.random.gamma(100., 1. / 100., (batchD, self._K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)

        sstats = n.zeros(self._lambda.shape)
        # Now, for each document d update that document's gamma and phi
        it = 0
        meanchange = 0
        for d in range(0, batchD):
            # These are mostly just shorthand (but might help cache locality)
            ids = list(wordids[d])
            cts = list(wordcts[d])
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self._expElogbeta[:, ids]
            # The optimal phi_{dwk} is proportional to
            # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
            # Iterate between gamma and phi until convergence
            for it in range(0, iter):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self._alpha + expElogthetad * \
                                       n.dot(cts / phinorm, expElogbetad.T)
                # print(gammad[:, n.newaxis])
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = n.exp(Elogthetad)
                phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
                # If gamma hasn't changed much, we're done.
                meanchange = n.mean(abs(gammad - lastgamma))
                if (meanchange < meanchangethresh):
                    break
            gamma[d, :] = gammad
            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            sstats[:, ids] += n.outer(expElogthetad.T, cts / phinorm)

        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk}
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        sstats = sstats * self._expElogbeta

        return ((gamma, sstats))

    def do_e_step_docs(self, docs):
        """
        Given a mini-batch of documents, estimates the parameters
        gamma controlling the variational distribution over the topic
        weights for each document in the mini-batch.

        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.

        Returns a tuple containing the estimated values of gamma,
        as well as sufficient statistics needed to update lambda.
        """
        # This is to handle the case where someone just hands us a single
        # document, not in a list.
        if (type(docs).__name__ == 'string'):
            temp = list()
            temp.append(docs)
            docs = temp

        (wordids, wordcts) = parse_doc_list(docs, self._vocab)

        return self.do_e_step(wordids, wordcts)

    def update_lambda_docs(self, docs, iter=50):
        """
        First does an E step on the mini-batch given in wordids and
        wordcts, then uses the result of that E step to update the
        variational parameter matrix lambda.

        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.

        Returns gamma, the parameters to the variational distribution
        over the topic weights theta for the documents analyzed in this
        update.

        Also returns an estimate of the variational bound for the
        entire corpus for the OLD setting of lambda based on the
        documents passed in. This can be used as a (possibly very
        noisy) estimate of held-out likelihood.
        """

        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._rhot = rhot
        # Do an E step to update gamma, phi | lambda for this
        # mini-batch. This also returns the information about phi that
        # we need to update lambda.
        (gamma, sstats) = self.do_e_step_docs(docs)
        # Estimate held-out likelihood for current values of lambda.
        bound = self.approx_bound_docs(docs, gamma)

        # Update lambda based on documents.
        self._lambda = self._lambda * (1 - rhot) + \
                       rhot * (self._eta + sstats)
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)

        self._updatect += 1
        # bound = self.approx_bound_docs(docs, gamma)
        return (gamma, bound)

    def update_lambda(self, wordids, wordcts):
        """
        First does an E step on the mini-batch given in wordids and
        wordcts, then uses the result of that E step to update the
        variational parameter matrix lambda.

        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.

        Returns gamma, the parameters to the variational distribution
        over the topic weights theta for the documents analyzed in this
        update.

        Also returns an estimate of the variational bound for the
        entire corpus for the OLD setting of lambda based on the
        documents passed in. This can be used as a (possibly very
        noisy) estimate of held-out likelihood.
        """

        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._rhot = rhot
        # Do an E step to update gamma, phi | lambda for this
        # mini-batch. This also returns the information about phi that
        # we need to update lambda.
        (gamma, sstats) = self.do_e_step(wordids, wordcts)
        # Estimate held-out likelihood for current values of lambda.
        bound = self.approx_bound(wordids, wordcts, gamma)

        # Update lambda based on documents.
        self._lambda = self._lambda * (1 - rhot) + \
                       rhot * (self._eta + sstats)
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        self._updatect += 1
        (gamma, sstats) = self.do_e_step(wordids, wordcts)
        return (gamma, bound)

    def approx_bound(self, wordids, wordcts, gamma):
        """
        Estimates the variational bound over *all documents* using only
        the documents passed in as "docs." gamma is the set of parameters
        to the variational distribution q(theta) corresponding to the
        set of documents passed in.

        The output of this function is going to be noisy, but can be
        useful for assessing convergence.
        """

        # This is to handle the case where someone just hands us a single
        # document, not in a list.
        batchD = len(wordids)

        score = 0
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)

        # E[log p(docs | theta, beta)]
        for d in range(0, batchD):
            gammad = gamma[d, :]
            ids = wordids[d]
            cts = n.array(wordcts[d])
            phinorm = n.zeros(len(ids))
            for i in range(0, len(ids)):
                temp = Elogtheta[d, :] + self._Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = n.log(sum(n.exp(temp - tmax))) + tmax
            score += n.sum(cts * phinorm)

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += n.sum((self._alpha - gamma) * Elogtheta)
        score += n.sum(gammaln(gamma) - gammaln(self._alpha))
        score += sum(gammaln(self._alpha * self._K) - gammaln(n.sum(gamma, 1)))

        # Compensate for the subsampling of the population of documents
        score = score

        # E[log p(beta | eta) - log q (beta | lambda)]
        score = score + n.sum((self._eta - self._lambda) * self._Elogbeta)
        score = score + n.sum(gammaln(self._lambda) - gammaln(self._eta))
        score = score + n.sum(gammaln(self._eta * self._W) -
                              gammaln(n.sum(self._lambda, 1)))

        return (score)

    def approx_bound_docs(self, docs, gamma):
        """
        Estimates the variational bound over *all documents* using only
        the documents passed in as "docs." gamma is the set of parameters
        to the variational distribution q(theta) corresponding to the
        set of documents passed in.

        The output of this function is going to be noisy, but can be
        useful for assessing convergence.
        """

        # This is to handle the case where someone just hands us a single
        # document, not in a list.
        if (type(docs).__name__ == 'string'):
            temp = list()
            temp.append(docs)
            docs = temp

        (wordids, wordcts) = parse_doc_list(docs, self._vocab)
        batchD = len(docs)

        score = 0
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)

        # E[log p(docs | theta, beta)]
        for d in range(0, batchD):
            gammad = gamma[d, :]
            ids = list(wordids[d])
            cts = n.array(list(wordcts[d]))
            phinorm = n.zeros(len(ids))
            for i in range(0, len(ids)):
                temp = Elogtheta[d, :] + self._Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = n.log(sum(n.exp(temp - tmax))) + tmax
            score += n.sum(cts * phinorm)
        # oldphinorm = phinorm
        #             phinorm = n.dot(expElogtheta[d, :], self._expElogbeta[:, ids])
        #             print oldphinorm
        #             print n.log(phinorm)
        #             score += n.sum(cts * n.log(phinorm))

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += n.sum((self._alpha - gamma) * Elogtheta)
        score += n.sum(gammaln(gamma) - gammaln(self._alpha))
        score += sum(gammaln(self._alpha * self._K) - gammaln(n.sum(gamma, 1)))

        # Compensate for the subsampling of the population of documents
        score = score

        # E[log p(beta | eta) - log q (beta | lambda)]
        score = score + n.sum((self._eta - self._lambda) * self._Elogbeta)
        score = score + n.sum(gammaln(self._lambda) - gammaln(self._eta))
        score = score + n.sum(gammaln(self._eta * self._W) -
                              gammaln(n.sum(self._lambda, 1)))

        return (score)


def classification_lda(K, data=NewsDataset(), perform_class=True, train_index=None, test_index=None):
    if train_index is None:
        doc_set_train = data.doc_set_train
        y_train_full = list(data.y_train)
    else:
        doc_set_train = [" ".join(d) for d in data.vectorizer.inverse_transform(data.X[train_index])]
        y_train_full = list(data.targets[train_index])

    if test_index is None:
        doc_set_test = data.doc_set_test
        y_test_full = list(data.y_test)
    else:
        doc_set_test = [" ".join(d) for d in data.vectorizer.inverse_transform(data.X[test_index])]
        y_test_full = list(data.targets[test_index])

    lda = OnlineLDA(vocab=data.vocabulary, K=K, alpha=1 / K, eta=1 / K, tau0=0, kappa=0)
    max_iterations = 100
    old_log_likelihood = n.finfo(n.float32).min

    for i in range(max_iterations):
        gamma_train, bound = lda.update_lambda_docs(list(filter(None, doc_set_train)))
        word_ids, word_cts = parse_doc_list(data.doc_set_train, lda._vocab)
        # estimate perpexity with the current batch
        perplexity = bound / sum(map(sum, word_cts))
        convergence = n.abs((bound - old_log_likelihood) / old_log_likelihood)
        if convergence < 1e-03:
            print('Converged after %d iterations, final log-likelihood: %.4f, final perplexity: %.4f'
                  % (i + 1, bound, perplexity))
            break
        old_log_likelihood = bound
        # print('iteration = %d: perplexity estimate = %f' % (i, perplexity))

    gamma_test, bound = lda.update_lambda_docs(list(filter(None, doc_set_test)))
    word_ids, word_cts = parse_doc_list(doc_set_test, lda._vocab)
    perplexity = bound / sum(map(sum, word_cts))
    print('heldout perplexity estimate = %f' % perplexity)

    y_train = list()
    for i, doc in enumerate(doc_set_train):
        if doc != '':
            y_train.append(y_train_full[i])
    y_test = list()
    for i, doc in enumerate(doc_set_test):
        if doc != '':
            y_test.append(y_test_full[i])

    gamma_test /= gamma_test.sum(axis=1)[:, n.newaxis]
    gamma_train /= gamma_train.sum(axis=1)[:, n.newaxis]

    if perform_class:
        for C in range(1, 16):
            print(C)
            clf = svm.SVC(C=C, random_state=0)
            clf.fit(gamma_train, y_train)
            print("Accuracy", clf.score(gamma_test, y_test))

    return gamma_train, gamma_test, y_train, y_test


def cross_validation(K):
    dataset = NewsDataset()
    accuracies_split = []
    print("number of topics", K)
    for train_index, test_index in dataset.splits:
        print("================= new split =================")
        gamma_train, gamma_test, y_train, y_test = classification_lda(K, data=dataset, perform_class=False,
                                                                      train_index=train_index, test_index=test_index)
        accuracies = []
        for C in range(1, 16):
            print("C", C)
            clf = svm.SVC(C=C, random_state=0)
            clf.fit(gamma_train, y_train)
            accuracy = clf.score(gamma_test, y_test)
            print("Accuracy", accuracy)
            accuracies.append(accuracy)
        best = max(accuracies)
        accuracies_split.append(best)
        print("---------Best Accuracy---------", best)
    print("============== Cross validation accuracy ============", accuracies_split)
    arr_accuracy = n.array(accuracies_split)
    print("Mean", n.mean(arr_accuracy))
    print("STD", n.std(arr_accuracy))


def perplexity_lda(K, dataset, train_sizes, random_state=10):
    print("-----------------LDA-----------------------")
    print("Number of topics", K)
    perplexities_tr = list()
    perplexities_te = list()
    max_iterations = 100
    for size in train_sizes:
        data = dataset(train_size=size, random_state=random_state)
        print("===============Observed words==============", size)
        lda = OnlineLDA(vocab=data.vocabulary, K=K, alpha=1 / K, eta=1 / K, tau0=0, kappa=0)
        old_log_likelihood = n.finfo(n.float32).min
        for i in range(max_iterations):
            gamma_train, bound = lda.update_lambda_docs(list(filter(None, data.doc_set_train)))
            word_ids, word_cts = parse_doc_list(data.doc_set_train, lda._vocab)
            # estimate training perplexity with the current batch
            perplexity = bound / sum(map(sum, word_cts))
            convergence = n.abs((bound - old_log_likelihood) / old_log_likelihood)
            if convergence < 1e-03:
                print('Converged after %d iterations, final log-likelihood: %.4f, final perplexity: %.4f'
                      % (i + 1, bound, perplexity))
                break
            old_log_likelihood = bound
            # print('iteration = %d: perplexity estimate = %f' % (i, perplexity))

        perplexities_tr.append(perplexity)
        gamma_test, bound_test = lda.update_lambda_docs(list(filter(None, data.doc_set_test)))
        word_ids, word_cts = parse_doc_list(data.doc_set_test, lda._vocab)
        perplexity_test = bound_test / sum(map(sum, word_cts))
        perplexities_te.append(perplexity_test)
        print('heldout perplexity estimate = %f' % perplexity_test)

    print("TRAIN SIZE: ", train_sizes)
    print("Train LDA: ", perplexities_tr)
    print("Test LDA: ", perplexities_te)

    return perplexities_tr, perplexities_te


def perplexity_lda_topics(topic_numbers, train_size, dataset):
    print("Train size ", train_size)
    perplexities_tr = list()
    perplexities_te = list()
    for t in topic_numbers:
        print("===============Number of topics==============", t)
        tr, te = perplexity_lda(train_sizes=[train_size], dataset=dataset, K=t)
        perplexities_tr.append(tr[0])
        perplexities_te.append(te[0])

    print("TOPICS: ", topic_numbers)
    print("Train LDA: ", perplexities_tr)
    print("Test LDA ", perplexities_te)


if __name__ == "__main__":
    d = ApDataset
    print("DATASET", d.__name__)

    k = 5
    train = n.linspace(0.1, 0.5, num=9)
    perplexity_lda(K=k, dataset=d, train_sizes=train)

    topics = [5, 8, 10, 15, 20, 25, 30, 35, 40]
    train_size = 0.9
    perplexity_lda_topics(topic_numbers=topics, train_size=train_size, dataset=d)

    classification_lda(K=15)
    cross_validation(50)
