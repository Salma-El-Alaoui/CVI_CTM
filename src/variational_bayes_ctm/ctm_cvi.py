import numpy as np
import time
from scipy.misc import logsumexp
import scipy as sp
from src.variational_bayes_ctm.ctm import CTM, compute_dirichlet_expectation


class CTM_CVI(CTM):
    def __init__(self, corpus, vocab, number_of_topics, alpha_mu=None, alpha_sigma=None, alpha_beta=None,
                 scipy_optimization_method="L-BFGS-B", em_max_iter=100, em_convergence=1e-03, local_param_iter=50,
                 step_size=0.7):
        super().__init__(corpus=corpus, vocab=vocab, number_of_topics=number_of_topics, alpha_mu=alpha_mu,
                         alpha_sigma=alpha_sigma, alpha_beta=alpha_beta,
                         scipy_optimization_method=scipy_optimization_method,
                         em_max_iter=em_max_iter, em_convergence=em_convergence, local_param_iter=local_param_iter)
        self.step_size = step_size

    def cvi_gaussian_update(self, doc_lambda, doc_nu_square, doc_nat_param_1, doc_nat_param_2, *args):

        (doc_zeta_factor, sum_phi, total_word_count, step_size) = args

        if self._diagonal_covariance_matrix:
            nat_param_1 = self._alpha_mu / self._alpha_sigma
        else:
            nat_param_1 = np.dot(self._alpha_mu, self._alpha_sigma_inv)[0, :]
        nat_param_1 += sum_phi

        exp_over_doc_zeta = logsumexp(doc_zeta_factor - doc_lambda[:, np.newaxis] - 0.5 * doc_nu_square[:, np.newaxis],
                                      axis=1)
        exp_over_doc_zeta = np.exp(-exp_over_doc_zeta)

        if self._diagonal_covariance_matrix:
            nat_param_2 = -0.5 * 1 / self._alpha_sigma
        else:
            nat_param_2 = -0.5 * np.diag(self._alpha_sigma_inv)
        nat_param_2 -= total_word_count * exp_over_doc_zeta

        assert nat_param_1.shape == (self._number_of_topics,)
        assert nat_param_2.shape == (self._number_of_topics,)

        new_doc_nat_param_1 = step_size * nat_param_1 + (1 - step_size) * nat_param_2
        new_doc_nat_param_2 = step_size * nat_param_2 + (1 - step_size) * nat_param_2

        new_doc_lambda = -1 * (new_doc_nat_param_1 / new_doc_nat_param_2)
        new_doc_nu_square = -1 / (2 * new_doc_nat_param_2)

        return new_doc_lambda, new_doc_nu_square, new_doc_nat_param_1, new_doc_nat_param_2

    def e_step(self, corpus=None):

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
        lambda_values = np.zeros((number_of_documents, self._number_of_topics))  # + self._alpha_mu[np.newaxis, :];
        nu_square_values = np.ones((number_of_documents, self._number_of_topics))  # + self._alpha_sigma[np.newaxis, :];
        # CVI
        nat_param_1_values = np.zeros((number_of_documents, self._number_of_topics))
        nat_param_2_values = np.zeros((number_of_documents, self._number_of_topics))

        # iterate over all documents
        for doc_id in range(number_of_documents):  # np.random.permutation
            # print("Document", doc_id)
            # initialize gamma for this document
            doc_lambda = lambda_values[doc_id, :]
            doc_nu_square = nu_square_values[doc_id, :]
            doc_nat_param_1 = nat_param_1_values[doc_id, :]
            doc_nat_param_2 = nat_param_2_values[doc_id, :]

            term_ids = word_ids[doc_id]
            term_counts = word_cts[doc_id]
            # compute the total number of words
            doc_word_count = np.sum(word_cts[doc_id])

            # update zeta in close form
            # doc_zeta = np.sum(np.exp(doc_lambda+0.5*doc_nu_square));
            doc_zeta_factor = doc_lambda + 0.5 * doc_nu_square
            doc_zeta_factor = np.tile(doc_zeta_factor, (self._number_of_topics, 1))

            for local_parameter_iteration_index in range(self._local_param_iter):
                # update phi in close form
                log_phi = E_log_eta[:, term_ids] + doc_lambda[:, np.newaxis]
                log_phi -= logsumexp(log_phi, axis=0)[np.newaxis, :]

                vb_updates = False
                if vb_updates:
                    # update lambda
                    sum_phi = np.exp(logsumexp(log_phi + np.log(term_counts), axis=1))
                    arguments = (doc_nu_square, doc_zeta_factor, sum_phi, doc_word_count)
                    doc_lambda = super().optimize_doc_lambda(doc_lambda, arguments)

                    # update zeta in close form
                    # doc_zeta = np.sum(np.exp(doc_lambda+0.5*doc_nu_square))
                    doc_zeta_factor = doc_lambda + 0.5 * doc_nu_square
                    doc_zeta_factor = np.tile(doc_zeta_factor, (self._number_of_topics, 1))

                    # update nu_square
                    arguments = (doc_lambda, doc_zeta_factor, doc_word_count)
                    # doc_nu_square = self.optimize_doc_nu_square(doc_nu_square, arguments)
                    doc_nu_square = super().optimize_doc_nu_square_in_log_space(doc_nu_square, arguments)

                # CVI
                # update lambda and nu square
                sum_phi = np.exp(logsumexp(log_phi + np.log(term_counts), axis=1))
                arguments = (doc_zeta_factor, sum_phi, doc_word_count, self.step_size)
                doc_lambda, doc_nu_square, doc_nat_param_1, doc_nat_param_2 = \
                    self.cvi_gaussian_update(doc_lambda, doc_nu_square, doc_nat_param_1, doc_nat_param_2, *arguments)

                # update zeta in close form
                doc_zeta_factor = doc_lambda + 0.5 * doc_nu_square
                doc_zeta_factor = np.tile(doc_zeta_factor, (self._number_of_topics, 1))

            if self._diagonal_covariance_matrix:
                document_log_likelihood -= 0.5 * np.sum(np.log(self._alpha_sigma))
                document_log_likelihood -= 0.5 * np.sum(doc_nu_square / self._alpha_sigma)
                document_log_likelihood -= 0.5 * np.sum((doc_lambda - self._alpha_mu) ** 2 / self._alpha_sigma)
            else:
                # TODO : replace logdet
                # TODO : replace by EPS
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

            # all terms including E_q[p(\eta | \beta)], i.e. terms involving \Psi(\eta)
            # are cancelled due to \eta updates in M-step

            lambda_values[doc_id, :] = doc_lambda
            nu_square_values[doc_id, :] = doc_nu_square

            # CVI
            nat_param_1_values[doc_id, :] = doc_nat_param_1
            nat_param_2_values[doc_id, :] = doc_nat_param_2

            phi_sufficient_statistics[:, term_ids] += np.exp(log_phi + np.log(term_counts))

        if corpus is None:
            self._lambda = lambda_values
            self._nu_square = nu_square_values
            return document_log_likelihood, phi_sufficient_statistics
        else:
            return words_log_likelihood, lambda_values, nu_square_values

    def fit(self):
        word_cts = self._parsed_corpus[1]
        normalizer = sum([np.sum(a) for a in word_cts])
        old_log_likelihood = np.finfo(np.float32).min
        convergences = list()
        for i in range(self._em_max_iter):
            log_likelihood, time = super().em_step()
            perplexity = log_likelihood / normalizer
            convergence = np.abs((log_likelihood - old_log_likelihood) / old_log_likelihood)
            convergences.append(convergence)
            if len(convergences) >= 2:
                av_conv = np.mean(np.asarray(convergences[-2:]))
            else:
                av_conv = np.mean(np.asarray(convergences))
            if av_conv < self._em_convergence:
                print('Converged after %d iterations, final log-likelihood: %.4f, final perplexity: %.4f'
                      % (i + 1, log_likelihood, perplexity))
                break
            old_log_likelihood = log_likelihood
            print('iteration: %d, log-likelihood: %.4f, log-perplexity: %.4f, convergence: %.4f'
                  % (i + 1, log_likelihood, perplexity, av_conv))
        return log_likelihood, perplexity

    def predict(self, test_corpus):
        parsed_corpus = super().parse_data(test_corpus)
        normalizer = sum([np.sum(a) for a in parsed_corpus[1]])
        clock_e_step = time.process_time()
        document_log_likelihood, lambda_values, nu_square_values = self.e_step(corpus=parsed_corpus)
        clock_e_step = time.process_time() - clock_e_step
        perplexity = document_log_likelihood / normalizer
        print('heldout log-likelihood: %.4f, heldout log-perplexity: %.4f' % (document_log_likelihood, perplexity))
        return document_log_likelihood, perplexity, lambda_values, nu_square_values

    def fit_predict(self, test_corpus):
        parsed_corpus_test = super().parse_data(test_corpus)
        normalizer_test = sum([np.sum(a) for a in parsed_corpus_test[1]])

        word_cts = self._parsed_corpus[1]
        normalizer = sum([np.sum(a) for a in word_cts])
        old_log_likelihood = np.finfo(np.float32).min

        lls_train = list()
        lls_test = list()
        times = list()

        for i in range(self._em_max_iter):
            log_likelihood, time = super().em_step()
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
