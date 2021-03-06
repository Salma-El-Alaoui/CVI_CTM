"""
Batch Variational Inference for the Correlated Topic Model
This code was modified from the code originally written by Ke Zhai.
Implements the Variational Inference algorithm for CTM as described in (Blei and Lafferty, 2007)
"""

import numpy as np
import time
from scipy.special import psi, gammaln
from scipy.misc import logsumexp
import scipy as sp


def compute_dirichlet_expectation(dirichlet_parameter):
    if len(dirichlet_parameter.shape) == 1:
        return psi(dirichlet_parameter) - psi(np.sum(dirichlet_parameter))
    return psi(dirichlet_parameter) - psi(np.sum(dirichlet_parameter, 1))[:, np.newaxis]


class CTM:
    """
    Implements the Variational Inference algorithm for CTM as described in (Blei and Lafferty, 2007)
    """

    def __init__(self, corpus, vocab, number_of_topics, alpha_mu=None, alpha_sigma=None, alpha_beta=None,
                 scipy_optimization_method="L-BFGS-B", em_max_iter=100, em_convergence=1e-03, local_param_iter=50):

        self.parse_vocabulary(vocab)
        # initialize the size of the vocabulary, i.e. total number of distinct tokens.
        self._number_of_types = len(self._type_to_index)

        self._corpus = corpus
        self._parsed_corpus = self.parse_data()
        # define the total number of document
        self._number_of_documents = len(self._parsed_corpus[0])

        # initialize the total number of topics.
        self._number_of_topics = number_of_topics

        if alpha_mu is None:
            alpha_mu = 0
        if alpha_sigma is None:
            alpha_sigma = 1
        if alpha_beta is None:
            alpha_beta = 1 / self._number_of_topics

        self._diagonal_covariance_matrix = False

        if self._diagonal_covariance_matrix:
            self._alpha_mu = np.zeros(self._number_of_topics) + alpha_mu
            self._alpha_sigma = np.zeros(self._number_of_topics) + alpha_sigma
        else:
            self._alpha_mu = np.zeros((1, self._number_of_topics)) + alpha_mu
            self._alpha_sigma = np.eye(self._number_of_topics) * alpha_sigma
            self._alpha_sigma_inv = np.linalg.pinv(self._alpha_sigma)

        self._alpha_beta = np.zeros(self._number_of_types) + alpha_beta

        self.init_latent_vars()
        self._counter = 0
        self._scipy_optimization_method = scipy_optimization_method
        self._em_max_iter = em_max_iter
        self._em_convergence = em_convergence
        self._local_param_iter = local_param_iter

    def parse_vocabulary(self, vocab):
        self._type_to_index = {}
        self._index_to_type = {}
        for word in set(vocab):
            self._index_to_type[len(self._index_to_type)] = word
            self._type_to_index[word] = len(self._type_to_index)
        self._vocab = self._type_to_index.keys()

    def parse_data(self, corpus=None):
        if corpus is None:
            corpus = self._corpus
        doc_count = 0

        word_ids = []
        word_cts = []

        for document_line in corpus:
            document_word_dict = {}
            for token in document_line.split():
                if token not in self._type_to_index:
                    continue
                type_id = self._type_to_index[token]
                if type_id not in document_word_dict:
                    document_word_dict[type_id] = 0
                document_word_dict[type_id] += 1

            if len(document_word_dict) == 0:
                continue

            word_ids.append(np.array(list(document_word_dict.keys())))
            word_cts.append(np.array(list(document_word_dict.values()))[np.newaxis, :])

            doc_count += 1

        print("Parsed %d Documents..." % doc_count)

        return word_ids, word_cts

    def init_latent_vars(self):
        # initialize a D-by-K matrix gamma
        self._lambda = np.zeros((self._number_of_documents, self._number_of_topics))
        self._nu_square = np.ones((self._number_of_documents, self._number_of_topics))
        # initialize a V-by-K matrix beta, subject to the sum over every row is 1
        self._eta = np.random.gamma(100., 0.01, (self._number_of_topics, self._number_of_types))

    """=================================================================================================================
    Helpers for optimization of the mean and variance of topic proportions
    ================================================================================================================="""

    def optimize_doc_lambda(self, doc_lambda, arguments):

        optimize_result = sp.optimize.minimize(self.f_doc_lambda,
                                               doc_lambda,
                                               args=arguments,
                                               method=self._scipy_optimization_method,
                                               jac=self.f_prime_doc_lambda,
                                               # hess=self.f_hessian_doc_lambda,
                                               # hessp=self.f_hessian_direction_doc_lambda,
                                               bounds=None,
                                               constraints=(),
                                               tol=None,
                                               callback=None,
                                               options={'disp': False, 'gtol': 1e-06})
        return optimize_result.x

    def f_doc_lambda(self, doc_lambda, *args):
        (doc_nu_square, doc_zeta_factor, sum_phi, total_word_count) = args

        exp_over_doc_zeta = logsumexp(doc_zeta_factor - doc_lambda[:, np.newaxis] - 0.5 * doc_nu_square[:, np.newaxis],
                                      axis=1)
        exp_over_doc_zeta = np.exp(-exp_over_doc_zeta)

        function_doc_lambda = np.sum(sum_phi * doc_lambda)

        if self._diagonal_covariance_matrix:
            mean_adjustment = doc_lambda - self._alpha_mu
            function_doc_lambda += -0.5 * np.sum((mean_adjustment ** 2) / self._alpha_sigma)
        else:
            mean_adjustment = doc_lambda[np.newaxis, :] - self._alpha_mu
            function_doc_lambda += -0.5 * np.dot(np.dot(mean_adjustment, self._alpha_sigma_inv), mean_adjustment.T)

        function_doc_lambda += -total_word_count * np.sum(exp_over_doc_zeta)

        return np.asscalar(-function_doc_lambda)

    def f_prime_doc_lambda(self, doc_lambda, *args):
        (doc_nu_square, doc_zeta_factor, sum_phi, total_word_count) = args

        exp_over_doc_zeta = logsumexp(doc_zeta_factor - doc_lambda[:, np.newaxis] - 0.5 * doc_nu_square[:, np.newaxis],
                                      axis=1)
        exp_over_doc_zeta = np.exp(-exp_over_doc_zeta)

        if self._diagonal_covariance_matrix:
            function_prime_doc_lambda = (self._alpha_mu - doc_lambda) / self._alpha_sigma
        else:
            function_prime_doc_lambda = np.dot((self._alpha_mu - doc_lambda[np.newaxis, :]), self._alpha_sigma_inv)[0,
                                        :]

        function_prime_doc_lambda += sum_phi
        function_prime_doc_lambda -= total_word_count * exp_over_doc_zeta

        return np.asarray(-function_prime_doc_lambda)

    def optimize_doc_nu_square_in_log_space(self, doc_nu_square, arguments, method_name=None):
        log_doc_nu_square = np.log(doc_nu_square)
        optimize_result = sp.optimize.minimize(self.f_log_doc_nu_square,
                                               log_doc_nu_square,
                                               args=arguments,
                                               method=method_name,
                                               jac=self.f_prime_log_doc_nu_square,
                                               # hess=self.f_hessian_log_doc_nu_square,
                                               # hessp=self.f_hessian_direction_log_doc_nu_square,
                                               bounds=None,
                                               constraints=(),
                                               tol=None,
                                               callback=None,
                                               options={'disp': False, 'gtol': 1e-06})

        log_doc_nu_square_update = optimize_result.x

        return np.exp(log_doc_nu_square_update)

    def f_doc_nu_square(self, doc_nu_square, *args):
        (doc_lambda, doc_zeta_factor, total_word_count) = args

        exp_over_doc_zeta = logsumexp(doc_zeta_factor - doc_lambda[:, np.newaxis] - 0.5 * doc_nu_square[:, np.newaxis],
                                      axis=1)
        exp_over_doc_zeta = np.exp(-exp_over_doc_zeta)

        function_doc_nu_square = 0.5 * np.sum(np.log(doc_nu_square))

        if self._diagonal_covariance_matrix:
            function_doc_nu_square += -0.5 * np.sum(doc_nu_square / self._alpha_sigma)
        else:
            function_doc_nu_square += -0.5 * np.sum(doc_nu_square * np.diag(self._alpha_sigma_inv))

        function_doc_nu_square += -total_word_count * np.sum(exp_over_doc_zeta)

        return np.asscalar(-function_doc_nu_square)

    def f_log_doc_nu_square(self, log_doc_nu_square, *args):
        return self.f_doc_nu_square(np.exp(log_doc_nu_square), *args)

    def f_prime_log_doc_nu_square(self, log_doc_nu_square, *args):
        (doc_lambda, doc_zeta_factor, total_word_count) = args

        exp_log_doc_nu_square = np.exp(log_doc_nu_square)

        exp_over_doc_zeta = sp.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, np.newaxis] - 0.5 * exp_log_doc_nu_square[:, np.newaxis], axis=1)
        exp_over_doc_zeta = np.exp(-exp_over_doc_zeta)

        if self._diagonal_covariance_matrix:
            function_prime_log_doc_nu_square = -0.5 * exp_log_doc_nu_square / self._alpha_sigma
        else:
            function_prime_log_doc_nu_square = -0.5 * exp_log_doc_nu_square * np.diag(self._alpha_sigma_inv)
        function_prime_log_doc_nu_square += 0.5
        function_prime_log_doc_nu_square -= 0.5 * total_word_count * exp_over_doc_zeta * exp_log_doc_nu_square

        return np.asarray(-function_prime_log_doc_nu_square)

    """=================================================================================================================
        E-step and M-step of the Variational Inference algorithm
    ================================================================================================================="""

    def m_step(self, phi_suff_stats):
        """
        M-step: update the variational parameter for topics
        """
        # compute the beta terms
        topic_log_likelihood = self._number_of_topics * \
                               (sp.special.gammaln(np.sum(self._alpha_beta)) - np.sum(gammaln(self._alpha_beta)))
        # compute the eta terms
        topic_log_likelihood += np.sum(np.sum(gammaln(self._eta), axis=1) - gammaln(np.sum(self._eta, axis=1)))

        self._eta = phi_suff_stats + self._alpha_beta
        return topic_log_likelihood

    def em_step(self):
        """
        Perform EM-update for one iteration and compute the training log-likelihood
        """
        self._counter += 1
        clock_e_step = time.process_time()
        document_log_likelihood, phi_sufficient_statistics = self.e_step()
        clock_e_step = time.process_time() - clock_e_step

        clock_m_step = time.process_time()
        topic_log_likelihood = self.m_step(phi_sufficient_statistics)
        clock_m_step = time.process_time() - clock_m_step

        joint_log_likelihood = document_log_likelihood + topic_log_likelihood
        # print(" E step  of iteration %d finished in %g seconds " % (self._counter, clock_e_step))
        # print(" M step of iteration %d finished in %g seconds" % (self._counter, clock_e_step))
        total_time = clock_e_step + clock_m_step
        return joint_log_likelihood[0][0], total_time

    def e_step(self, corpus=None):
        """
        E-step: update the variational parameters for topic proportions and topic assignments.
        """
        if corpus is None:
            word_ids = self._parsed_corpus[0]
            word_cts = self._parsed_corpus[1]
        else:
            word_ids = corpus[0]
            word_cts = corpus[1]

        number_of_documents = len(word_ids)

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

        # iterate over all documents
        for doc_id in range(number_of_documents):  # np.random.permutation
            # initialize gamma for this document
            doc_lambda = lambda_values[doc_id, :]
            doc_nu_square = nu_square_values[doc_id, :]

            term_ids = word_ids[doc_id]
            term_counts = word_cts[doc_id]
            # compute the total number of words
            doc_word_count = np.sum(word_cts[doc_id])

            # update zeta in close form
            # doc_zeta = np.sum(np.exp(doc_lambda+0.5*doc_nu_square))
            doc_zeta_factor = doc_lambda + 0.5 * doc_nu_square
            doc_zeta_factor = np.tile(doc_zeta_factor, (self._number_of_topics, 1))

            for local_parameter_iteration_index in range(self._local_param_iter):
                # update phi in close form
                log_phi = E_log_eta[:, term_ids] + doc_lambda[:, np.newaxis]
                log_phi -= logsumexp(log_phi, axis=0)[np.newaxis, :]

                # update lambda
                sum_phi = np.exp(logsumexp(log_phi + np.log(term_counts), axis=1))
                arguments = (doc_nu_square, doc_zeta_factor, sum_phi, doc_word_count)
                doc_lambda = self.optimize_doc_lambda(doc_lambda, arguments)

                # update zeta in closed form
                # doc_zeta = np.sum(np.exp(doc_lambda+0.5*doc_nu_square))
                doc_zeta_factor = doc_lambda + 0.5 * doc_nu_square
                doc_zeta_factor = np.tile(doc_zeta_factor, (self._number_of_topics, 1))

                # update nu_square
                arguments = (doc_lambda, doc_zeta_factor, doc_word_count)
                # doc_nu_square = self.optimize_doc_nu_square(doc_nu_square, arguments)
                doc_nu_square = self.optimize_doc_nu_square_in_log_space(doc_nu_square, arguments)

                # update zeta in close form
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

            if corpus is not None:
                # compute the phi terms
                words_log_likelihood += np.sum(np.exp(log_phi + np.log(term_counts)) * E_log_prob_eta[:, term_ids])

            # all terms including E_q[p(\eta | \beta)], i.e., terms involving \Psi(\eta),
            # are cancelled due to \eta updates in M-step

            lambda_values[doc_id, :] = doc_lambda
            nu_square_values[doc_id, :] = doc_nu_square

            phi_sufficient_statistics[:, term_ids] += np.exp(log_phi + np.log(term_counts))

        if corpus is None:
            self._lambda = lambda_values
            self._nu_square = nu_square_values
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
        normalizer = sum([np.sum(a) for a in word_cts])
        old_log_likelihood = np.finfo(np.float32).min
        for i in range(self._em_max_iter):
            log_likelihood = self.em_step()
            perplexity = log_likelihood / normalizer
            convergence = np.abs((log_likelihood - old_log_likelihood) / old_log_likelihood)
            if convergence < self._em_convergence:
                print('Converged after %d iterations, final log-likelihood: %.4f, final perplexity: %.4f'
                      % (i + 1, log_likelihood, perplexity))
                break
            old_log_likelihood = log_likelihood
            print('iteration: %d, log-likelihood: %.4f, log-perplexity: %.4f, convergence: %.4f'
                  % (i + 1, log_likelihood, perplexity, convergence))
        return log_likelihood, perplexity

    def predict(self, test_corpus):
        """
        Performs E-step on test corpus using stored topics obtained by training
        Computes the average heldout log-likelihood
        """
        parsed_corpus = self.parse_data(test_corpus)
        normalizer = sum([np.sum(a) for a in parsed_corpus[1]])
        clock_e_step = time.time()
        document_log_likelihood, lambda_values, nu_square_values = self.e_step(corpus=parsed_corpus)
        clock_e_step = time.time() - clock_e_step
        perplexity = document_log_likelihood / normalizer
        print('heldout log-likelihood: %.4f, heldout log-perplexity: %.4f' % (document_log_likelihood, perplexity))
        return document_log_likelihood, perplexity, lambda_values, nu_square_values

    def fit_predict(self, test_corpus):
        """
        Computes the heldout-log likelihood on the test corpus after every iteration of training.
        """
        parsed_corpus_test = self.parse_data(test_corpus)
        normalizer_test = sum([np.sum(a) for a in parsed_corpus_test[1]])

        word_cts = self._parsed_corpus[1]
        normalizer = sum([np.sum(a) for a in word_cts])
        old_log_likelihood = np.finfo(np.float32).min

        lls_train = list()
        lls_test = list()
        times = list()

        for i in range(self._em_max_iter):
            log_likelihood, time = self.em_step()
            perplexity = log_likelihood / normalizer
            convergence = np.abs((log_likelihood - old_log_likelihood) / old_log_likelihood)
            times.append(time)
            lls_train.append(perplexity)
            old_log_likelihood = log_likelihood
            print('iteration: %d, log-likelihood: %.4f, log-perplexity: %.4f, convergence: %.4f, time: %.4f'
                  % (i + 1, log_likelihood, perplexity, convergence, time))

            document_log_likelihood_test, _, _ = self.e_step(corpus=parsed_corpus_test)
            perplexity_test = document_log_likelihood_test / normalizer_test
            lls_test.append(perplexity_test)
            print('heldout log-likelihood: %.4f, heldout log-perplexity: %.4f' % (document_log_likelihood_test,
                                                                                  perplexity_test))

        return lls_train, lls_test, times

    def export_beta(self, exp_beta_path, top_display=-1):
        """
        Writes the "top_display" more likely words in each topic and their probability
        """
        output = open(exp_beta_path, 'w')
        E_log_eta = compute_dirichlet_expectation(self._eta)
        for topic_index in range(self._number_of_topics):
            output.write("==========\t%d\t==========\n" % (topic_index))
            beta_probability = np.exp(E_log_eta[topic_index, :] - sp.misc.logsumexp(E_log_eta[topic_index, :]))
            i = 0
            for type_index in reversed(np.argsort(beta_probability)):
                i += 1
                output.write("%s\t%g\n" % (self._index_to_type[type_index], beta_probability[type_index]))
                if top_display > 0 and i >= top_display:
                    break
        output.close()
