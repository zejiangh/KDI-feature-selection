import math
import tensorflow as tf

class Kernel(object):

    def __call__(self, X):

        pass

class LinearKernel(Kernel):

    def __call__(self, X):
        return tf.matmul(X, X, transpose_b=True)

class PolynomialKernel(Kernel):

    def __init__(self, a, b, d):
        self.a = a
        self.b = b
        self.d = d

    def __call__(self, X):
        return (self.a * tf.matmul(X, X, transpose_b=True) + self.b) ** self.d

class LaplacianKernel(Kernel):

    def __init__(self, sigma):
        assert sigma > 0
        self.sigma = sigma

    def __call__(self, X):
        X_rowdiff = tf.expand_dims(X, 1) - tf.expand_dims(X, 0)
        return tf.exp(-tf.reduce_sum(tf.abs(X_rowdiff), 2) / self.sigma)

class GaussianKernel(Kernel):

    def __init__(self, sigma):
        assert sigma > 0
        self.sigma = sigma

    def __call__(self, X):
        X_inner = tf.matmul(X, tf.transpose(X))
        X_norm = tf.diag_part(X_inner)
        X_dist_sq = X_norm+tf.reshape(X_norm, [-1,1])-2*X_inner
        return tf.exp(-X_dist_sq/(2*self.sigma**2))

class EqualityKernel(Kernel):

    def __init__(self, composition="product"):
        assert composition in ("mean", "product")
        self.composition = composition

    def __call__(self, X):
        X_equal = tf.to_double(tf.equal(tf.expand_dims(X, 0), tf.expand_dims(X, 1)))
        reduce = {
            "mean": tf.reduce_mean,
            "product": tf.reduce_prod
        }[self.composition]
        return reduce(X_equal, reduction_indices=2)
