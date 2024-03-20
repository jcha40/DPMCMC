import tensorflow as tf
import tensorflow_probability as tfp

# Input functions for multinomial items

def join_cluster_multinomial(feature_shape=None, dtype=None):
    def join_cluster(item, cluster_tensor):
        return tfp.distributions.DirichletMultinomial(tf.reduce_sum(item), cluster_tensor + 1.).log_prob(item)
    return join_cluster

def new_cluster_multinomial(feature_shape=None, dtype=None):
    def new_cluster(item):
        return tfp.distributions.DirichletMultinomial(tf.reduce_sum(item), tf.ones(feature_shape[0], dtype=dtype)).log_prob(item)
    return new_cluster

def cluster_prior_multinomial(feature_shape=None, dtype=None, max_clusters=None):
    def cluster_prior(data, current_state):
        return tf.math.unsorted_segment_sum(data, current_state, max_clusters)
    return cluster_prior

# Input functions for nonparametric log-probability mass items

def join_cluster_dot_product(feature_shape=None, dtype=None):
    def join_cluster(item, cluster_tensor):
        return tf.reduce_sum(tf.reduce_logsumexp(cluster_tensor + item, axis=-1), axis=range(1, len(feature_shape)))
    return join_cluster

def new_cluster_dot_product(feature_shape=None, dtype=None):
    def new_cluster(item):
        return tf.reduce_sum(tf.reduce_logsumexp(item, axis=-1) - tf.math.log(tf.cast(feature_shape[-1], dtype)))
    return new_cluster

def cluster_prior_dot_product(feature_shape=None, dtype=None, max_clusters=None):
    def cluster_prior(data, current_state):
        prior = tf.math.unsorted_segment_sum(data, current_state, max_clusters)
        return prior - tf.reduce_logsumexp(prior, axis=1, keepdims=True)
    return cluster_prior
