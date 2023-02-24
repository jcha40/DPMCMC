# DPMCMC

This is a Python implementation of the Dirichlet Process Monte Carlo Markov Chain algorithm described in [1]. The code is based on the implementation by PhylogicNDT [2] (https://github.com/broadinstitute/PhylogicNDT).

### Example 1
```python
from DPMCMC import DotProductKernel, run_dpmcmc
import numpy as np
import tensorflow as tf
import scipy.stats

# Generate some data
np.random.seed(0)
tf.random.set_seed(0)
# x-values for beta distributions
x = np.expand_dims(np.linspace(.05, .95, 100), (0, 1))
# 3 clusters each with 5 beta distributed components and 100 samples [100 samples, 5 distributions, 100 x-values]
c1 = scipy.stats.beta.pdf(x, scipy.stats.norm.rvs(loc=np.expand_dims([71, 21, 51, 61, 61], (0, 2)), scale=2, size=(100, 5, 1)),
                          scipy.stats.norm.rvs(loc=np.expand_dims([31, 81, 51, 41, 41], (0, 2)), scale=2, size=(100, 5, 1)))
c2 = scipy.stats.beta.pdf(x, scipy.stats.norm.rvs(loc=np.expand_dims([41, 51, 31, 21, 11], (0, 2)), scale=2, size=(100, 5, 1)),
                          scipy.stats.norm.rvs(loc=np.expand_dims([61, 51, 71, 81, 91], (0, 2)), scale=2, size=(100, 5, 1)))
c3 = scipy.stats.beta.pdf(x, scipy.stats.norm.rvs(loc=np.expand_dims([11, 31, 51, 71, 91], (0, 2)), scale=2, size=(100, 5, 1)),
                          scipy.stats.norm.rvs(loc=np.expand_dims([91, 71, 51, 31, 11], (0, 2)), scale=2, size=(100, 5, 1)))

# Concatenate and normalize [300 samples, 5 distributions, 100 x-values]
data = np.concatenate([c1, c2, c3], axis=0) + 1e-20
data = data / np.sum(data, axis=2, keepdims=True)
data = tf.constant(np.log(data))

# Run DPMCMC
dp_engine = DotProductKernel(data)
trace = run_dpmcmc(dp_engine, 100)
```

### Example 2
```python
from DPMCMC import MultinomialKernel, run_dpmcmc
import numpy as np
import tensorflow as tf
import scipy.stats

# Generate some data
np.random.seed(0)
tf.random.set_seed(0)

# 3 clusters each drawing 50 items into 6 categories for 100 samples [100 samples, 6 categories]
c1 = tf.convert_to_tensor(scipy.stats.multinomial.rvs(50, [.1, .1, .1, .5, .1, .1], size=100), dtype=tf.float64)
c2 = tf.convert_to_tensor(scipy.stats.multinomial.rvs(50, [.7, .1, .05, .05, .05, .05], size=100), dtype=tf.float64)
c3 = tf.convert_to_tensor(scipy.stats.multinomial.rvs(50, [.01, .04, .55, .1, .2, .1], size=100), dtype=tf.float64)

# Concatenate [300 samples, 6 categories]
data = tf.concat([c1, c2, c3], axis=0)

# Run DPMCMC
dp_engine = MultinomialKernel(data)
trace = run_dpmcmc(dp_engine, 100)
```

[1] Escobar, M.D. and West, M. (1995) “Bayesian density estimation and inference using mixtures,” Journal of the American Statistical Association, 90(430), pp. 577–588. Available at: https://doi.org/10.1080/01621459.1995.10476550.

[2] Leshchiner, I. et al. (2018) “Comprehensive analysis of tumour initiation, spatial and temporal progression under multiple lines of treatment.” Available at: https://doi.org/10.1101/508127.