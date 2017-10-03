"""
Stochastic Variational Inference for the Correlated Topic Model.
"""

import numpy as np
import time
from scipy.special import psi, gammaln
from scipy.misc import logsumexp
import scipy as sp
from src.variational_bayes_ctm.ctm_cvi import CTM_CVI
from src.variational_bayes_ctm.ctm_cvi import compute_dirichlet_expectation


def gen_batches(n, batch_size):
    start = 0
    list_ranges = list()
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        list_ranges.append(list(range(start, end)))
        start = end
    if start < n:
        list_ranges.append(list(range(start, n)))
    return list_ranges


class CTM_CVI_S(CTM_CVI):
    """
    Implements Stochastic Variational Inference for the Correlated Topic Model
    """

    def __init__(self, corpus, vocab, number_of_topics, alpha_mu=None, alpha_sigma=None, alpha_beta=None,
                 scipy_optimization_method="L-BFGS-B", em_max_iter=100, em_convergence=1e-03, step_size=0.7,
                 local_param_iter=50, batch_size=-1, learning_offset=10, learning_decay=0.7, evaluate_every=1):
        super().__init__(corpus=corpus, vocab=vocab, number_of_topics=number_of_topics, alpha_mu=alpha_mu,
                         alpha_sigma=alpha_sigma, alpha_beta=alpha_beta,
                         scipy_optimization_method=scipy_optimization_method,
                         em_max_iter=em_max_iter, em_convergence=em_convergence, local_param_iter=local_param_iter,
                         step_size=step_size)
        self._batch_size = batch_size
        self._tau0 = learning_offset
        self._kappa = learning_decay
        self._evaluate_every = evaluate_every

    """=================================================================================================================
            E-step and M-step of the Variational Inference algorithm
    ================================================================================================================="""

    def m_step_s(self, phi_suff_stats, batch_indexes):
        """
        Stochastic M-step: update the variational parameter for topics using a mini-batch of documents
        """
        word_ids = self._parsed_corpus[0]
        number_of_documents = len(word_ids)
        batch_size = len(batch_indexes)
        # compute the beta terms
        topic_log_likelihood = self._number_of_topics * \
                               (sp.special.gammaln(np.sum(self._alpha_beta)) - np.sum(gammaln(self._alpha_beta)))
        # compute the eta terms
        topic_log_likelihood += np.sum(np.sum(gammaln(self._eta), axis=1) - gammaln(np.sum(self._eta, axis=1)))
        # self._eta = phi_suff_stats + self._alpha_beta
        rhot = np.power(self._tau0 + self._counter, -self._kappa)
        self._eta = self._eta * (1 - rhot) + \
                    rhot * (self._alpha_beta + number_of_documents * phi_suff_stats / batch_size)

        return topic_log_likelihood

    def em_step_s(self, batch_indexes):
        """
        Performs stochastic EM-update for one iteration using a mini-batch of documents
        and compute the training log-likelihood
        """
        self._counter += 1
        clock_e_step = time.process_time()
        document_log_likelihood, phi_sufficient_statistics = self.e_step(batch_indexes=batch_indexes, corpus=None)
        clock_e_step = time.process_time() - clock_e_step

        clock_m_step = time.process_time()
        topic_log_likelihood = self.m_step_s(phi_suff_stats=phi_sufficient_statistics, batch_indexes=batch_indexes)
        clock_m_step = time.process_time() - clock_m_step

        joint_log_likelihood = document_log_likelihood + topic_log_likelihood
        # print(" E step  of iteration %d finished in %g seconds " % (self._counter, clock_e_step))
        # print(" M step of iteration %d finished in %g seconds" % (self._counter, clock_e_step))
        total_time = clock_e_step + clock_m_step
        return joint_log_likelihood[0][0], total_time

    def e_step(self, batch_indexes=None, corpus=None):
        """
        E-step: update the variational parameters for topic proportions and topic assignments.
        """
        if corpus is None:
            word_ids = self._parsed_corpus[0]
            word_cts = self._parsed_corpus[1]
        else:
            word_ids = corpus[0]
            word_cts = corpus[1]
            batch_indexes = range(len(word_ids))

        number_of_documents = len(batch_indexes)
        total_number_documents = len(word_ids)

        E_log_eta = compute_dirichlet_expectation(self._eta)

        if corpus is not None:
            E_log_prob_eta = E_log_eta - sp.misc.logsumexp(E_log_eta, axis=1)[:, np.newaxis]

        document_log_likelihood = 0
        words_log_likelihood = 0

        # initialize a V_matrix-by-K matrix phi sufficient statistics
        phi_sufficient_statistics = np.zeros((self._number_of_topics, self._number_of_types))

        # initialize a D-by-K matrix lambda and nu_square values
        lambda_values = np.zeros((number_of_documents, self._number_of_topics))
        nu_square_values = np.ones((number_of_documents, self._number_of_topics))
        # CVI
        nat_param_1_values = np.zeros((number_of_documents, self._number_of_topics))
        nat_param_2_values = np.zeros((number_of_documents, self._number_of_topics))

        # iterate over all documents
        for doc_id_set, doc_id in enumerate(batch_indexes):  # np.random.permutation
            # initialize gamma for this document
            doc_lambda = lambda_values[doc_id_set, :]
            doc_nu_square = nu_square_values[doc_id_set, :]
            doc_nat_param_1 = nat_param_1_values[doc_id_set, :]
            doc_nat_param_2 = nat_param_2_values[doc_id_set, :]

            term_ids = word_ids[doc_id]
            term_counts = word_cts[doc_id]
            # compute the total number of words
            doc_word_count = np.sum(word_cts[doc_id])

            # update zeta in close form
            # doc_zeta = np.sum(np.exp(doc_lambda+0.5*doc_nu_square));
            doc_zeta_factor = doc_lambda + 0.5 * doc_nu_square
            doc_zeta_factor = np.tile(doc_zeta_factor, (self._number_of_topics, 1))

            for local_parameter_iteration_index in range(self._local_param_iter):
                # update phi in closed form
                log_phi = E_log_eta[:, term_ids] + doc_lambda[:, np.newaxis]
                log_phi -= logsumexp(log_phi, axis=0)[np.newaxis, :]

                # CVI
                # update lambda and nu square
                sum_phi = np.exp(logsumexp(log_phi + np.log(term_counts), axis=1))
                arguments = (doc_zeta_factor, sum_phi, doc_word_count, self.step_size)
                doc_lambda, doc_nu_square, doc_nat_param_1, doc_nat_param_2 = \
                    super().cvi_gaussian_update(doc_lambda, doc_nu_square, doc_nat_param_1, doc_nat_param_2, *arguments)

                # update zeta in closed form
                doc_zeta_factor = doc_lambda + 0.5 * doc_nu_square
                doc_zeta_factor = np.tile(doc_zeta_factor, (self._number_of_topics, 1))

            if self._diagonal_covariance_matrix:
                document_log_likelihood -= 0.5 * np.sum(np.log(self._alpha_sigma))
                document_log_likelihood -= 0.5 * np.sum(doc_nu_square / self._alpha_sigma)
                document_log_likelihood -= 0.5 * np.sum((doc_lambda - self._alpha_mu) ** 2 / self._alpha_sigma)
            else:
                document_log_likelihood -= 0.5 * np.log(np.linalg.det(self._alpha_sigma) + 1e-30)
                document_log_likelihood -= 0.5 * np.sum(doc_nu_square * np.diag(self._alpha_sigma_inv))
                document_log_likelihood -= 0.5 * np.dot(
                    np.dot((self._alpha_mu - doc_lambda[np.newaxis, :]), self._alpha_sigma_inv),
                    (self._alpha_mu - doc_lambda[np.newaxis, :]).T)

            document_log_likelihood += np.sum(np.sum(np.exp(log_phi) * term_counts, axis=1) * doc_lambda)
            # use the fact that doc_zeta = np.sum(np.exp(doc_lambda+0.5*doc_nu_square)), to cancel the factors
            document_log_likelihood -= logsumexp(doc_lambda + 0.5 * doc_nu_square) * doc_word_count

            document_log_likelihood += 0.5 * self._number_of_topics

            document_log_likelihood += 0.5 * np.sum(np.log(doc_nu_square))

            document_log_likelihood -= np.sum(np.exp(log_phi) * log_phi * term_counts)

            # print('\t document-log-likelihood: %.4f' % document_log_likelihood)
            if corpus is not None:
                # compute the phi terms
                words_log_likelihood += np.sum(np.exp(log_phi + np.log(term_counts)) * E_log_prob_eta[:, term_ids])

            # all terms including E_q[p(\eta | \beta)], i.e., terms involving \Psi(\eta),
            # are cancelled due to \eta updates in M-step

            lambda_values[doc_id_set, :] = doc_lambda
            nu_square_values[doc_id_set, :] = doc_nu_square

            # CVI
            nat_param_1_values[doc_id_set, :] = doc_nat_param_1
            nat_param_2_values[doc_id_set, :] = doc_nat_param_2

            phi_sufficient_statistics[:, term_ids] += np.exp(log_phi + np.log(term_counts))

        if corpus is None:
            self._lambda = lambda_values
            self._nu_square = nu_square_values
            document_log_likelihood = document_log_likelihood * total_number_documents / number_of_documents
            return document_log_likelihood, phi_sufficient_statistics
        else:
            return words_log_likelihood, lambda_values, nu_square_values

    """=================================================================================================================
        Training and testing
    ================================================================================================================="""

    def fit(self):
        """
        Performs EM-update until reaching target average change in the log-likelihood
        """
        word_cts = self._parsed_corpus[1]
        word_ids = self._parsed_corpus[0]
        normalizer = sum([np.sum(a) for a in word_cts])
        old_log_likelihood = np.finfo(np.float32).min
        convergences = list()

        number_of_documents = len(word_ids)
        if self._batch_size == -1:
            batch_size = number_of_documents
        else:
            batch_size = self._batch_size

        for i in range(self._em_max_iter):
            for docs_idx in gen_batches(number_of_documents, batch_size):
                log_likelihood, time = self.em_step_s(batch_indexes=docs_idx)
                perplexity = log_likelihood / normalizer
                convergence = np.abs((log_likelihood - old_log_likelihood) / old_log_likelihood)
                convergences.append(convergence)
                if len(convergences) >= 1:
                    av_conv = np.mean(np.asarray(convergences[-1:]))
                else:
                    av_conv = np.mean(np.asarray(convergences))
                if av_conv < self._em_convergence:
                    print('Converged after %d epochs, final log-likelihood: %.4f, final perplexity: %.4f'
                          % (i + 1, log_likelihood, perplexity))
                    break
                old_log_likelihood = log_likelihood
                print('epoch: %d, log-likelihood: %.4f, log-perplexity: %.4f, convergence: %.4f'
                      % (i + 1, log_likelihood, perplexity, convergence))
        return log_likelihood, perplexity

    def predict(self, test_corpus):
        """
        Performs E-step on test corpus using stored topics obtained by training
        Computes the average heldout log-likelihood
        """
        parsed_corpus = super().parse_data(test_corpus)
        normalizer = sum([np.sum(a) for a in parsed_corpus[1]])
        clock_e_step = time.process_time()
        document_log_likelihood, lambda_values, nu_square_values = self.e_step(corpus=parsed_corpus)
        clock_e_step = time.process_time() - clock_e_step
        perplexity = document_log_likelihood / normalizer
        print('heldout log-likelihood: %.4f, heldout log-perplexity: %.4f' % (document_log_likelihood, perplexity))
        return document_log_likelihood, perplexity, lambda_values, nu_square_values

    def fit_predict(self, test_corpus):
        """
        Computes the heldout-log likelihood on the test corpus after "evaluate_every" iterations
        (mini-batches) of training.
        """
        parsed_corpus_test = super().parse_data(test_corpus)
        normalizer_test = sum([np.sum(a) for a in parsed_corpus_test[1]])

        word_cts = self._parsed_corpus[1]
        word_ids = self._parsed_corpus[0]
        normalizer = sum([np.sum(a) for a in word_cts])
        old_log_likelihood = np.finfo(np.float32).min
        convergences = list()

        number_of_documents = len(word_ids)
        if self._batch_size == -1:
            batch_size = number_of_documents
        else:
            batch_size = self._batch_size

        lls_train = list()
        lls_test = list()
        for epoch in range(self._em_max_iter):
            print('epoch: %d' % (epoch + 1))
            epoch_train = list()
            epoch_test = list()
            batches = gen_batches(number_of_documents, batch_size)
            nb_mini_batches = len(batches)
            for i, docs_idx in enumerate(batches):
                log_likelihood, time = self.em_step_s(batch_indexes=docs_idx)
                perplexity = log_likelihood / normalizer
                convergence = np.abs((log_likelihood - old_log_likelihood) / old_log_likelihood)
                convergences.append(convergence)
                old_log_likelihood = log_likelihood
                print('mini-batch: %d / %d, log-likelihood: %.4f, log-perplexity: %.4f, convergence: %.4f'
                      % (i + 1, nb_mini_batches, log_likelihood, perplexity, convergence))
                epoch_train.append(perplexity)
                if (i + 1) % self._evaluate_every == 0 and (i + 1) != nb_mini_batches:
                    document_log_likelihood_test, _, _ = self.e_step(corpus=parsed_corpus_test)
                    perplexity_test = document_log_likelihood_test / normalizer_test
                    epoch_test.append(perplexity_test)
                    print('heldout log-likelihood: %.4f, heldout log-perplexity: %.4f' % (document_log_likelihood_test,
                                                                                          perplexity_test))
            # evaluate the log-likelihood at the end of the batch either way
            document_log_likelihood_test, _, _ = self.e_step(corpus=parsed_corpus_test)
            perplexity_test = document_log_likelihood_test / normalizer_test
            epoch_test.append(perplexity_test)
            print('heldout log-likelihood full pass for epoch %d : %.4f, heldout log-perplexity: %.4f'
                  % ((epoch + 1), document_log_likelihood_test, perplexity_test))
            lls_train.append(epoch_train)
            lls_test.append(epoch_test)

        return lls_train, lls_test
