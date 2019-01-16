from __future__ import print_function
import numpy as np
import tensorflow as tf
import kernel
def center(X):

    mean_col = tf.reduce_mean(X, axis=0, keep_dims=True)
    mean_row = tf.reduce_mean(X, axis=1, keep_dims=True)
    mean_all = tf.reduce_mean(X)
    return X - mean_col - mean_row + mean_all

def project(v, z):
    """[1] http://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf"""

    assert v.ndim == 1
    assert z > 0

    z = float(z)

    mu = np.sort(v)[::-1]
    mu_cumsum = np.cumsum(mu)
    max_index = np.nonzero(mu * np.arange(1, len(v) + 1) > mu_cumsum - z)[0][-1]
    theta = (mu_cumsum[max_index] - z) / (max_index + 1)
    return np.maximum(v - theta, 0)

class DI(object):
    def __init__(
            self,
            X,
            Y, 
            transform_Y,
            epsilon,
            mu,
            D_approx = None
    ): 

        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        assert X.ndim == 2
        assert Y.ndim == 1

        n, d = X.shape
        assert Y.shape == (n,)
        self.d = d
        self.n = n

        # Whitening transform for X.
        X = (X - X.mean(axis=0)) / (X.std(axis=0)+1e-20)
        if n>1000:
            idx = np.random.choice(n, 100)
            tmp = np.take(X, idx, axis=0)
            sigma = np.median(np.linalg.norm(tmp[:, None, :] - tmp[None, :, :], axis=2))
        else:
            sigma = np.median(np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2))
        kernel_X = kernel.GaussianKernel(sigma)
            
#        kernel_X1 = kernel.GaussianKernel(1)
#        kernel_X2 = kernel.GaussianKernel(2)
#        kernel_X3 = kernel.GaussianKernel(4)
#        kernel_X4 = kernel.GaussianKernel(8)
#        kernel_X5 = kernel.GaussianKernel(16)

        # Transform Y.
        assert transform_Y in (None, "binary", "one-hot")

        if transform_Y == "binary":
            values = sorted(set(Y.ravel()))
            assert len(values) == 2
            Y_new = np.zeros(n)
            Y_new[Y == values[0]] = -1
            Y_new[Y == values[1]] = 1
            Y = Y_new

        elif transform_Y == "one-hot":
            values = sorted(set(Y.ravel()))
            Y_new = np.zeros((n, len(values)))
            for i, value in enumerate(values):
                Y_new[Y == value, i] = 1
            Y = Y_new

        if Y.ndim == 1:
            Y = Y[:, np.newaxis]

        Y = Y - Y.mean(0)

        # Build graph for loss and gradients computation. 
        with tf.Graph().as_default():
            self.w = tf.placeholder(tf.float64, shape=d) 
            self.inputs = [self.w]

            if D_approx is None:
                
                raw_kernel = kernel_X(X*self.w)
                
#                raw_kernel = 1/5*(kernel_X1(X*self.w)+kernel_X2(X*self.w)+kernel_X3(X*self.w)+kernel_X4(X*self.w)+kernel_X5(X*self.w))
                
                G_X_w = center(raw_kernel)+1e-6*np.eye(n)
                
                Cov_Y = tf.matmul(Y, Y, transpose_b=True)
                
                G_B_w = tf.matmul(tf.matmul(G_X_w, Cov_Y), G_X_w)

                G_X_w = tf.matmul(G_X_w, G_X_w) + epsilon * G_X_w

                G_X_w_inv = tf.matrix_inverse(G_X_w)

            self.loss = tf.trace(tf.matmul(G_X_w_inv, G_B_w))-mu*tf.reduce_sum(tf.multiply(self.w, 1-self.w))

            self.gradients = tf.gradients(self.loss, self.inputs)

            self.sess = tf.Session()

    def solve_gradient_descent(self, num_features, learning_rate = 0.001, iterations = 1000, verbose=True):

        assert num_features <= self.d

        # Initialize w. 
#        w = project(np.random.rand(self.d), num_features)
        w = project(np.ones(self.d), num_features)
        inputs = [w]

        def clip_and_project(w):
            """ clip and project w onto the simplex."""
            w = w.clip(0, 1)
            if w.sum() > num_features:
                w = project(w, num_features)
            return w

        for iteration in range(1, iterations + 1):

            # Compute loss and print.
            if verbose:
                loss = self.sess.run(self.loss, feed_dict=dict(zip(self.inputs, inputs)))
                print("iteration {} loss {}".format(iteration, loss))

            # Update w with projected gradient method. 
            gradients = self.sess.run(self.gradients, feed_dict=dict(zip(self.inputs, inputs)))
            for i, gradient in enumerate(gradients):
                inputs[i] += learning_rate * gradient
            inputs[0] = clip_and_project(inputs[0])

        # Compute rank of each feature based on weight.
        # Random permutation to avoid bias due to equal weights.
        idx = np.random.permutation(self.d) 
        permutated_weights = inputs[0][idx]  
        permutated_ranks=(-permutated_weights).argsort().argsort()+1
        self.ranks = permutated_ranks[np.argsort(idx)]

        return inputs[0] 

def di(X, Y, num_features, type_Y, epsilon, mu, learning_rate = 0.001, 
    iterations = 1000, D_approx = None, verbose = True): 

    assert type_Y in ('ordinal','binary','categorical','real-valued')
    if type_Y == 'ordinal' or type_Y == 'real-valued':
        transform_Y = None 
    elif type_Y == 'binary':
        transform_Y = 'binary'
    elif type_Y == 'categorical':
        transform_Y = 'one-hot'

    fs = DI(X, Y, transform_Y, epsilon, mu, D_approx = D_approx)
    w = fs.solve_gradient_descent(num_features, learning_rate, iterations, verbose)
    if verbose:
        print('The weights on featurs are: ', w)
    ranks = fs.ranks 
    return ranks