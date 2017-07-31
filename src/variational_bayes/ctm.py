import numpy as np
import sys
import time
from scipy.special import psi
from scipy.misc import logsumexp
import scipy as sp

def compute_dirichlet_expectation(dirichlet_parameter):
    if (len(dirichlet_parameter.shape) == 1):
        return psi(dirichlet_parameter) - psi(np.sum(dirichlet_parameter))
    return psi(dirichlet_parameter) - psi(np.sum(dirichlet_parameter, 1))[:, np.newaxis]

class CTM:
    def __init__(self, corpus, vocab, number_of_topics, alpha_mu=None, alpha_sigma=None, alpha_beta=None, scipy_optimization_method="L-BFGS-B"):

        self.parse_vocabulary(vocab)
        # initialize the size of the vocabulary, i.e. total number of distinct tokens.
        self._number_of_types = len(self._type_to_index)

        self._corpus = corpus
        self._parsed_corpus = self.parse_data()
        # define the total number of document
        self._number_of_documents = len(self._parsed_corpus[0])

        # initialize the total number of topics.
        self._number_of_topics = number_of_topics

        if alpha_mu == None:
            alpha_mu = 0
        if alpha_sigma == None:
            alpha_sigma = 1/self._number_of_topics
        if alpha_beta == None:
            alpha_beta = 1/self._number_of_types

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

    def parse_vocabulary(self, vocab):
        self._type_to_index = {}
        self._index_to_type = {}
        for word in set(vocab):
            self._index_to_type[len(self._index_to_type)] = word
            self._type_to_index[word] = len(self._type_to_index)
        self._vocab = self._type_to_index.keys()

    def parse_data(self, corpus=None):
        if corpus == None:
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
                sys.stderr.write("problem occurred during parsing")
                continue

            word_ids.append(np.array(list(document_word_dict.keys())))
            word_cts.append(np.array(list(document_word_dict.values()))[np.newaxis, :])

            doc_count += 1
            if doc_count % 10000 == 0:
                print("Parsed %d Documents..." % doc_count)

        assert len(word_ids) == len(word_cts);
        print("Parsed %d Documents..." % (doc_count))

        return word_ids, word_cts

    def init_latent_vars(self):
        # initialize a D-by-K matrix gamma
        self._lambda = np.zeros((self._number_of_documents, self._number_of_topics))
        self._nu_square = np.ones((self._number_of_documents, self._number_of_topics))
        # initialize a V-by-K matrix beta, subject to the sum over every row is 1
        self._eta = np.random.gamma(100., 1. / 100., (self._number_of_topics, self._number_of_types))

    def optimize_doc_lambda(self, doc_lambda, arguments):

        optimize_result = sp.optimize.minimize(self.f_doc_lambda,
                                                  doc_lambda,
                                                  args=arguments,
                                                  method=self._scipy_optimization_method,
                                                  jac=self.f_prime_doc_lambda,
                                                  hess=self.f_hessian_doc_lambda,
                                                  hessp=self.f_hessian_direction_doc_lambda,
                                                  bounds=None,
                                                  constraints=(),
                                                  tol=None,
                                                  callback=None,
                                                  options={'disp': False})
        return optimize_result.x
    #TODO: compute differantial, hessian and hessian direction

    def em_step(self):
        self._counter += 1
        clock_e_step = time.time()
        document_log_likelihood, phi_sufficient_statistics = self.e_step()
        clock_e_step = time.time() - clock_e_step

        clock_m_step = time.time()
        topic_log_likelihood = self.m_step(phi_sufficient_statistics)
        clock_m_step = time.time() - clock_m_step

        print(document_log_likelihood, topic_log_likelihood)
        joint_log_likelihood = document_log_likelihood + topic_log_likelihood

        print(" E step  of iteration %d finished in %g seconds " % (self._counter, clock_e_step))
        print(" M step of iteration %d finished in %g seconds" % (self._counter, clock_e_step))
        print("log-likelihood: %g" % joint_log_likelihood)

    def e_step(self, local_parameter_iteration=10):
        word_ids = self._parsed_corpus[0]
        word_cts = self._parsed_corpus[1]

        number_of_documents = len(word_ids)

        E_log_eta = compute_dirichlet_expectation(self._eta)

        document_log_likelihood = 0
        words_log_likelihood = 0

        # initialize a V_matrix-by-K matrix phi sufficient statistics
        phi_sufficient_statistics = np.zeros((self._number_of_topics, self._number_of_types))

        # initialize a D-by-K matrix lambda and nu_square values
        lambda_values = np.zeros((number_of_documents, self._number_of_topics))  # + self._alpha_mu[np.newaxis, :];
        nu_square_values = np.ones((number_of_documents, self._number_of_topics))  # + self._alpha_sigma[np.newaxis, :];

        # iterate over all documents
        for doc_id in np.random.permutation(number_of_documents):
            # initialize gamma for this document
            doc_lambda = lambda_values[doc_id, :]
            doc_nu_square = nu_square_values[doc_id, :]

            term_ids = word_ids[doc_id]
            term_counts = word_cts[doc_id]
            # compute the total number of words
            doc_word_count = np.sum(word_cts[doc_id])

            # update zeta in close form
            # doc_zeta = np.sum(np.exp(doc_lambda+0.5*doc_nu_square));
            doc_zeta_factor = doc_lambda + 0.5 * doc_nu_square
            doc_zeta_factor = np.tile(doc_zeta_factor, (self._number_of_topics, 1))

            for local_parameter_iteration_index in range(local_parameter_iteration):
                # update phi in close form
                log_phi = E_log_eta[:, term_ids] + doc_lambda[:, np.newaxis]
                log_phi -= logsumexp(log_phi, axis=0)[np.newaxis, :]

                # update lambda
                sum_phi = np.exp(logsumexp(log_phi + np.log(term_counts), axis=1))
                arguments = (doc_nu_square, doc_zeta_factor, sum_phi, doc_word_count)
                doc_lambda = self.optimize_doc_lambda(doc_lambda, arguments)

                # update zeta in close form
                # doc_zeta = np.sum(np.exp(doc_lambda+0.5*doc_nu_square));
                doc_zeta_factor = doc_lambda + 0.5 * doc_nu_square
                doc_zeta_factor = np.tile(doc_zeta_factor, (self._number_of_topics, 1))

                # update nu_square
                arguments = (doc_lambda, doc_zeta_factor, doc_word_count)
                # doc_nu_square = self.optimize_doc_nu_square(doc_nu_square, arguments);
                doc_nu_square = self.optimize_doc_nu_square_in_log_space(doc_nu_square, arguments)
 
                # update zeta in close form
                # doc_zeta = np.sum(np.exp(doc_lambda+0.5*doc_nu_square));
                doc_zeta_factor = doc_lambda + 0.5 * doc_nu_square
                doc_zeta_factor = np.tile(doc_zeta_factor, (self._number_of_topics, 1))

            if self._diagonal_covariance_matrix:
                document_log_likelihood -= 0.5 * np.sum(np.log(self._alpha_sigma))
                document_log_likelihood -= 0.5 * np.sum(doc_nu_square / self._alpha_sigma)
                document_log_likelihood -= 0.5 * np.sum((doc_lambda - self._alpha_mu) ** 2 / self._alpha_sigma)
            else:
                #TODO : replace logdet
                #TODO : replace by EPS
                document_log_likelihood -= 0.5 * np.log(np.linalg.det(self._alpha_sigma) + 1e-30)
                document_log_likelihood -= 0.5 * np.sum(doc_nu_square * np.diag(self._alpha_sigma_inv))
                document_log_likelihood -= 0.5 * np.dot(
                    np.dot((self._alpha_mu - doc_lambda[np.newaxis, :]), self._alpha_sigma_inv),
                    (self._alpha_mu - doc_lambda[np.newaxis, :]).T)

            document_log_likelihood += np.sum(np.sum(np.exp(log_phi) * term_counts, axis=1) * doc_lambda)
            # use the fact that doc_zeta = np.sum(np.exp(doc_lambda+0.5*doc_nu_square)), to cancel the factors
            document_log_likelihood -= logsumexp(doc_lambda + 0.5 * doc_nu_square) * doc_word_count

            document_log_likelihood += 0.5 * self._number_of_topics;
            # document_log_likelihood += 0.5 * self._number_of_topics * np.log(2 * np.pi)
            document_log_likelihood += 0.5 * np.sum(np.log(doc_nu_square))

            document_log_likelihood -= np.sum(np.exp(log_phi) * log_phi * term_counts)

            # Note: all terms including E_q[p(\eta | \beta)], i.e., terms involving \Psi(\eta), are cancelled due to \eta updates in M-step

            lambda_values[doc_id, :] = doc_lambda
            nu_square_values[doc_id, :] = doc_nu_square

            phi_sufficient_statistics[:, term_ids] += np.exp(log_phi + np.log(term_counts))

            if (doc_id + 1) % 1000 == 0:
                print("successfully processed %d documents..." % (doc_id + 1))

            self._lambda = lambda_values
            self._nu_square = nu_square_values
            return document_log_likelihood, phi_sufficient_statistics

