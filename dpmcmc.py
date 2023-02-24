import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import scipy.stats
import scipy.special
import scipy.optimize
import math
import logging
import itertools
import collections
import input_functions

KernelResults = collections.namedtuple('KernelResults', [])

class DPMCMCKernel(tfp.mcmc.TransitionKernel):
    """
    Implements a Dirichlet process MCMC kernel as described by Escobar & West 1995.
    """
    def __init__(self, data, join_cluster, new_cluster, cluster_prior, max_clusters=30, r=3., mu=3., dtype=tf.float64, seed=None, verbose=True):
        """
        Args:
            data: Input data should be a tensor of rank 2 or higher with shape [n_items, ...].
            join_cluster: Wrapper function that takes the feature shape and dtype as arguments and returns a function
                that computes the log-likelihood of an item joining each cluster. The function should take two
                arguments, the item data and the tensor of cluster priors. The item data should be a tensor of
                rank 2 or higher with shape [1, ...]. The cluster priors should be a tensor of rank 2 or higher
                with shape [max_clusters, ...]. The function should return a tensor of rank 1 with shape
                [max_clusters].
            new_cluster: Wrapper function that takes the feature shape and dtype as arguments and returns a function
                that computes the log-likelihood of an item creating a new cluster. The function should take one
                argument, the item data. The item data should be a tensor of rank 2 or higher with shape [1, ...].
                The function should return a scalar.
            cluster_prior: Wrapper function that takes the feature shape, dtype, and max clusters as arguments and
                returns a function that computes the cluster priors. The function should take two arguments, the
                data and the current cluster assignments. The data should be a tensor of rank 2 or higher with
                shape [n_items, ...] (same as data input for __init__). The cluster assignments should be a tensor of
                rank 1 with shape [n_items]. The function should return a tensor of rank 2 or higher with shape
                [max_clusters, ...].
            max_clusters: Maximum number of clusters to allow.
            r: DP prior parameter.
            mu: DP prior parameter.
            dtype: Data type for computations.
            seed: Random seed.
            verbose: Print progress if True.
        """
        self.dtype = dtype
        self.data = tf.convert_to_tensor(data, dtype=self.dtype)
        shape = tf.shape(self.data).numpy()
        self.n_items = shape[0]
        self.feature_shape = tuple(shape[1:])
        self.max_clusters = max_clusters
        self.join_cluster = join_cluster(self.feature_shape, self.dtype)
        self.new_cluster = new_cluster(self.feature_shape, self.dtype)
        self.cluster_prior = cluster_prior(self.feature_shape, self.dtype, self.max_clusters)
        self.a, self.b = self.init_dp_prior(r, mu)
        self.alpha = scipy.stats.gamma.rvs(self.a, scale=1. / self.b)
        if seed is not None:
            np.random.set_state(np.random.RandomState(seed).get_state())
            tf.random.set_seed(seed)
        self.verbose = verbose

    def one_step(self, current_state, previous_kernel_results):
        """
            current_state: current cluster assignments
        """
        # loop over items with random order
        for i in np.random.choice(self.n_items, self.n_items, replace=False):
            item = self.data[i:i + 1]
            # remove current item from its current cluster
            current_state = tf.tensor_scatter_nd_update(current_state, [[i]], [-1])
            # compute number items in each cluster
            items_per_cluster = tf.reduce_sum(tf.cast(tf.expand_dims(current_state, 1) == tf.expand_dims(tf.range(self.max_clusters), 0), tf.int32), axis=0)
            # compute number of unoccupied clusters
            n_unoccupied_clusters = tf.reduce_sum(tf.cast(items_per_cluster == 0, tf.float64))
            # create priors for each cluster
            cluster_tensor = self.cluster_prior(self.data, current_state)
            # compute log-likelihoods of joining each cluster
            join_cluster_loglik = self.join_cluster(item, cluster_tensor)
            # compute log-likelihood of creating a new cluster
            new_cluster_loglik = self.new_cluster(item)
            # replace log-likelihoods of unoccupied clusters with log-likelihood of creating a new cluster
            loglik = tf.where(items_per_cluster > 0, join_cluster_loglik, new_cluster_loglik - tf.math.log(n_unoccupied_clusters))
            # coefficients from chinese restaurant process
            crp_coefs = tf.where(items_per_cluster > 0, tf.cast(items_per_cluster, tf.float64) / (self.n_items - 1 + self.alpha),
                                 self.alpha / (self.n_items - 1 + self.alpha))
            # update cluster assignment
            new_assignment = tf.random.categorical([loglik + tf.math.log(crp_coefs)], 1, dtype=tf.int32)[0]
            current_state = tf.tensor_scatter_nd_update(current_state, [[i]], new_assignment)

        n_clusters = tf.size(tf.unique(current_state)[0]).numpy()
        if self.verbose:
            print('k = {}'.format(n_clusters))

        # resample alpha as described by Escobar & West 1995
        eta = scipy.stats.beta.rvs(self.alpha + 1, self.n_items)
        self.alpha = self.sample_gamma_cond_N_k(n_clusters, eta)

        return current_state, previous_kernel_results

    def init_dp_prior(self, r, mu):
        k_prior = scipy.stats.nbinom.pmf(range(1, self.n_items + 1), r, r / (r + mu))
        k_prior = k_prior / sum(k_prior)

        logging.info('Initializing prior over DP k for ' + str(self.n_items) + ' items')

        log_stirling_coef = self.get_log_stirling_coefs(self.n_items)
        k_0_map = self.get_k_0_map(self.n_items, np.linspace(1e-25, 5, 1000), log_stirling_coef)

        return self.get_gamma_prior_from_k_prior(self.n_items, k_0_map, k_prior)

    @staticmethod
    def get_log_stirling_coefs(N):
        """
        Get the log of the unsigned Stirling numbers of the first kind using the recursive formula described here:
            https://en.wikipedia.org/wiki/Stirling_numbers_of_the_first_kind#Recurrence_relation
        """
        curr = np.zeros(1)
        for n in range(N):
            curr = np.logaddexp(np.log(n) + curr[1:], curr[:-1])  # n * [n k] + [n k-1]
            curr = np.insert(curr, [0, n], [-np.inf, 0])  # put 0 at the beginning and 1 at the end
        return curr

    @staticmethod
    def get_k_0_map(N, gamma_GRID, log_stirling_coef):
        # map 1st moments of (k | N, gamma), over a grid on gamma
        N_gamma = len(gamma_GRID)
        # (alpha^k) * s(n, k) * n! * gamma(alpha) / gamma(alpha + n)
        # values on gamma_GRID are alpha values
        # gamma refers to gamma function
        # Escobar and West 1995 for DP likelihood equation 10
        DP_prob = np.outer(np.log(gamma_GRID), np.arange(N)) + log_stirling_coef[1:] + math.lgamma(N + 1) + \
            np.reshape(scipy.special.gammaln(gamma_GRID) - scipy.special.gammaln(gamma_GRID + N), (N_gamma, 1))
        # normalize
        DP_prob = np.exp(DP_prob - np.reshape(scipy.special.logsumexp(DP_prob, axis=1), (N_gamma, 1)))
        # map expected k to alpha
        k_0_map = np.stack((np.sum(DP_prob * np.arange(1, N + 1), axis=1), gamma_GRID), axis=1)
        return k_0_map

    @staticmethod
    def get_gamma_prior_from_k_prior(N, k_0_map, k_prior):
        int_k_prior = np.interp(k_0_map[:, 0], range(1, N + 1), k_prior)
        int_k_prior = int_k_prior / float(np.sum(int_k_prior))

        def LL(Par):

            mu = Par[0]
            sigma = Par[1]
            if mu < 0 or sigma < 0:
                return np.inf

            B = mu / float(sigma ** 2)
            A = B * mu

            k_prob = scipy.stats.gamma.pdf(k_0_map[:, 1], A, scale=1. / B)
            diff = np.diff(k_0_map[:, 0])
            k_prob = k_prob / np.append(2 * diff[0] - diff[1], diff)
            k_prob = k_prob / np.sum(k_prob)

            return scipy.stats.entropy(k_prob, int_k_prior)

        kl_div = np.inf
        A = np.nan
        B = np.nan

        for par in itertools.product(range(1, 25, 5), range(1, 25, 5)):
            res = scipy.optimize.minimize(LL, par, method='Nelder-Mead')
            opt = res.x
            obj = LL(opt)
            if obj < kl_div:
                kl_div = obj
                B = opt[0] / (opt[1] ** 2)
                A = B * opt[0]

        return A, B

    def sample_gamma_cond_N_k(self, k, eta):
        m1 = scipy.stats.gamma.rvs(self.a + k, scale=1. / (self.b - np.log(eta)))
        m2 = scipy.stats.gamma.rvs(self.a + k - 1, scale=1. / (self.b - np.log(eta)))

        pi_eta_ratio = (self.a + k - 1) / (self.n_items * float(self.b - np.log(eta)))
        pi_eta = pi_eta_ratio / (pi_eta_ratio + 1)

        new_gamma = pi_eta * m1 + (1 - pi_eta) * m2

        return new_gamma

    @property
    def is_calibrated(self):
        return True

class MultinomialKernel(DPMCMCKernel):
    """
    Multinomial kernel for the Dirichlet process MCMC. Input data should be a tensor of rank 2 with the first dimension
    being the number of data points and the second dimension being the number of categories.
    """
    def __init__(self, data, max_clusters=30, r=3., mu=3., dtype=tf.float64, seed=None, verbose=True):
        super(MultinomialKernel, self).__init__(data, input_functions.join_cluster_multinomial,
            input_functions.new_cluster_multinomial, input_functions.cluster_prior_multinomial,
            max_clusters=max_clusters, r=r, mu=mu, dtype=dtype, seed=seed, verbose=verbose)

class DotProductKernel(DPMCMCKernel):
    """
    Dot product kernel for the Dirichlet process MCMC. Input data should be a tensor of rank 2 or higher with the first
    dimension being the number of data points and the last dimension being the axis along which to compute the dot
    product.
    """
    def __init__(self, data, max_clusters=30, r=3., mu=3., dtype=tf.float64, seed=None, verbose=True):
        super(DotProductKernel, self).__init__(data, input_functions.join_cluster_dot_product,
            input_functions.new_cluster_dot_product, input_functions.cluster_prior_dot_product,
            max_clusters=max_clusters, r=r, mu=mu, dtype=dtype, seed=seed, verbose=verbose)

def run_dpmcmc(kernel, n_iterations, initial_state=None):
    """
    Run the Dirichlet process MCMC.
    Args:
        kernel: DPMCMCKernel instance.
        n_iterations: Number of iterations.
        initial_state: Initial cluster assignments (default: all assigned to cluster 0).

    Returns:

    """
    if initial_state is None:
        initial_state = tf.zeros(kernel.n_items, dtype=tf.int32)

    return tfp.mcmc.sample_chain(n_iterations, initial_state, trace_fn=None, parallel_iterations=1, kernel=kernel, previous_kernel_results=KernelResults())
