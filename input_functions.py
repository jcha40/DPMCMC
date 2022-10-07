import tensorflow as tf
import tensorflow_probability as tfp


def cluster_prior_as_sum(self, current_state):
    return tf.math.unsorted_segment_sum(self.data, current_state, self.max_clusters)


# Input functions for multinomial items

@staticmethod
def join_cluster_multinomial(item, cluster_tensor):
    return tfp.distributions.DirichletMultinomial(tf.reduce_sum(item), cluster_tensor + 1.).log_prob(item)


def new_cluster_multinomial(self, item):
    return tfp.distributions.Multinomial(tf.reduce_sum(item), tf.ones(self.n_features, dtype=self.dtype)).log_prob(item)


# Input functions for nonparametric log-probability mass items

@staticmethod
def join_cluster_dot_product(item, cluster_tensor):
    return tf.reduce_logsumexp(cluster_tensor + item)


def new_cluster_dot_product(self, item):
    return tf.reduce_logsumexp(item - tf.math.log(self.n_features))
