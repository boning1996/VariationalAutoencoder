from __future__ import print_function
import tensorflow as tf
import numpy as np
from data import *

N_data, train_images, train_labels, test_images, test_labels = load_mnist()
train_images = np.round(train_images[0:10000])
train_labels = train_labels[0:10000]
test_images = np.round(test_images[0:10000])
train_batches = np.split(train_images, 100)
batch_size = 100
epoches = 200
latent_dimension = 2
unit_variance = tf.constant([1, 1], dtype=tf.float32)
unit_mean = tf.constant([0, 0], dtype=tf.float32)
# [-4, 4] x [-4, 4] grid for integration
grid = np.array([[[(i - 40)/10, (j - 40)/10] for j in range(80)] for i in range(80)])
grid = tf.reshape(tf.convert_to_tensor(grid, dtype=tf.float32), (6400, 2))

weights = {
    'w_xh': tf.Variable(tf.zeros((500, 784))),
    'w_hsigma': tf.Variable(tf.zeros((2, 500))),
    'w_hmu': tf.Variable(tf.zeros((2, 500))),
    'w_zh': tf.Variable(tf.zeros((500, 2))),
    'w_hy': tf.Variable(tf.zeros((784, 500))),
}

biases = {
    'b_xh': tf.Variable(tf.zeros(500)),
    'b_hsigma': tf.Variable(tf.zeros(2)),
    'b_hmu': tf.Variable(tf.zeros(2)),
    'b_zh': tf.Variable(tf.zeros(500)),
    'b_hy': tf.Variable(tf.zeros(784)),
}

x = tf.placeholder(tf.float32, shape=(None, 784))

# sample from Gaussian given mean and covariances(diagonal)
def gaussian_sample(size, dimension, mean, variance):
    samples = []
    for i in range(size):
        unit_random = tf.convert_to_tensor(np.random.randn(dimension), dtype=tf.float32)
        z = mean[i] + tf.multiply(unit_random, tf.sqrt(variance[i]))
        samples.append(z)
    return samples


# sample from Bernoulli given probabilities
def bernoulli_sample(size, dimension, y):
    rand = tf.convert_to_tensor(np.random.rand(size, dimension), dtype=tf.float32)
    samples = tf.cast(rand < y, tf.float32)
    return samples


# log-pdf of Gaussian
def log_pdf_gaussian(samples, dimension, mean, variance):
    return tf.constant(-dimension/2 * np.log(2*np.pi), dtype=tf.float32) \
           - 0.5*tf.log(tf.reduce_prod(variance, axis=-1)) \
           - 0.5*tf.reduce_sum(tf.multiply((samples - mean)**2, 1/variance), axis=-1)


# log-pdf of Bernoulli
def log_pdf_bernoulli(samples, y):
    return tf.reduce_sum(tf.multiply(samples, tf.log(y)) + tf.multiply(1 - samples, tf.log(1 - y)), axis=-1)


def log_sum_exp(values):
    return tf.reduce_logsumexp(values)


# integrate log_p_x|z + log_p_z on [-4, 4] x [-4, 4] grid spaced by 0.1
# sample_x: batch_size x 784
# z_grid: 6400 x 2
# y_grid: 6400 x 784
# returns p_x: batch_size x 1
def intergrate_p_x_z(sample_x, z_grid, y_grid):
    size = sample_x.shape[0]
    sample_x = tf.reshape(tf.tile(sample_x, [1, 6400]), (size, 6400, 784))
    z_grid = tf.reshape(tf.tile(z_grid, [1, size]), (size, 6400, latent_dimension))
    y_grid = tf.reshape(tf.tile(y_grid, [1, size]), (size, 6400, 784))
    log_pxz = log_pdf_bernoulli(sample_x, y_grid)
    log_pz = log_pdf_gaussian(z_grid, latent_dimension, unit_mean, unit_variance)
    return tf.reduce_logsumexp(log_pxz + log_pz, axis=1)


def integrate_check(sample_x, z_grid, y_grid, log_px):
    size = sample_x.shape[0]
    log_px = tf.reshape(log_px, (size, 1))
    log_px = tf.reshape(tf.tile(log_px, [1, 6400]), (size, 6400))
    sample_x = tf.reshape(tf.tile(sample_x, [1, 6400]), (size, 6400, 784))
    z_grid = tf.reshape(tf.tile(z_grid, [1, size]), (size, 6400, latent_dimension))
    y_grid = tf.reshape(tf.tile(y_grid, [1, size]), (size, 6400, 784))
    log_pxz = log_pdf_bernoulli(sample_x, y_grid)
    log_pz = log_pdf_gaussian(z_grid, latent_dimension, unit_mean, unit_variance)
    return tf.reduce_logsumexp(log_pxz + log_pz - log_px, axis=1)






def check(batch_x):
    # W_xh = tf.where(tf.is_nan(weights['w_xh']), tf.zeros_like(weights['w_xh']), weights['w_xh'])
    W_xh = weights['w_xh']
    W_hsigma = weights['w_hsigma']
    W_hmu = weights['w_hmu']
    W_zh = weights['w_zh']
    W_hy = weights['w_hy']
    b_xh = biases['b_xh']
    b_hsigma = biases['b_hsigma']
    b_hmu = biases['b_hmu']
    b_zh = biases['b_zh']
    b_hy = biases['b_hy']
    batch_x = tf.convert_to_tensor(batch_x, dtype=tf.float32)
    h = tf.tanh(tf.matmul(batch_x, tf.transpose(W_xh)) + b_xh)
    log_variance = tf.matmul(h, tf.transpose(W_hsigma)) + b_hsigma
    variances = tf.exp(log_variance)
    mu = tf.matmul(h, tf.transpose(W_hmu)) + b_hmu
    z = gaussian_sample(batch_size, latent_dimension, mu, variances)
    z = tf.convert_to_tensor(z)
    y = tf.sigmoid(tf.matmul(tf.tanh(tf.matmul(z, tf.transpose(W_zh)) + b_zh), tf.transpose(W_hy)) + b_hy)
    y = tf.clip_by_value(y, clip_value_min=0.0001, clip_value_max=0.9999)
    x_decoded = bernoulli_sample(batch_size, 784, y)
    log_q = log_pdf_gaussian(z, latent_dimension, mu, variances)
    log_pz = log_pdf_gaussian(z, latent_dimension, unit_mean, unit_variance)
    log_px_given_z = log_pdf_bernoulli(x_decoded, y)
    z_grid = grid
    y_grid = tf.sigmoid(tf.matmul(tf.tanh(tf.matmul(z_grid, tf.transpose(W_zh)) + b_zh), tf.transpose(W_hy)) + b_hy)
    log_px = intergrate_p_x_z(x_decoded, z_grid, y_grid)
    c = integrate_check(x_decoded, z_grid, y_grid, log_px)

    elbo = tf.reduce_sum(-log_q + log_pz + log_px_given_z, axis=0)/batch_size

    # c_log_p_x = log_sum_exp(intergrate_p_x_z(x_decoded[0], y[0]))
    # c_int = log_sum_exp(integrate_check(x_decoded[0], y[0], a))
    return [log_px, c]


def compute_elbo(batch_x):

    W_xh = weights['w_xh']
    W_hsigma = weights['w_hsigma']
    W_hmu = weights['w_hmu']
    W_zh = weights['w_zh']
    W_hy = weights['w_hy']
    b_xh = biases['b_xh']
    b_hsigma = biases['b_hsigma']
    b_hmu = biases['b_hmu']
    b_zh = biases['b_zh']
    b_hy = biases['b_hy']
    # batch_x = tf.convert_to_tensor(batch_x, dtype=tf.float32)
    # ecoding hidden layer
    h = tf.tanh(tf.matmul(batch_x, tf.transpose(W_xh)) + b_xh)
    log_variance = tf.matmul(h, tf.transpose(W_hsigma)) + b_hsigma
    variances = tf.exp(log_variance)
    mu = tf.matmul(h, tf.transpose(W_hmu)) + b_hmu
    # sample latent
    z = gaussian_sample(batch_size, latent_dimension, mu, variances)
    z = tf.convert_to_tensor(z)
    # decode to Bernoulli probabilities
    y = tf.sigmoid(tf.matmul(tf.tanh(tf.matmul(z, tf.transpose(W_zh)) + b_zh), tf.transpose(W_hy)) + b_hy)
    # y = tf.clip_by_value(y, clip_value_min=0.0001, clip_value_max=0.9999)
    # sample decoded x from Bernoulli
    x_decoded = bernoulli_sample(batch_size, 784, y)
    log_q = log_pdf_gaussian(z, latent_dimension, mu, variances)
    log_pz = log_pdf_gaussian(z, latent_dimension, unit_mean, unit_variance)
    log_px_given_z = log_pdf_bernoulli(x_decoded, y)
    elbo = tf.reduce_sum(-log_q + log_pz + log_px_given_z, axis=0)/batch_size

    return elbo


loss = -compute_elbo(x)
optimizer = tf.train.AdamOptimizer(learning_rate=0.2).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    i = 0
    # Keep training until reach max iterations
    while i < epoches*batch_size:
        # Run optimization op (backprop)
        batch_index = i % batch_size
        cur_batch = train_batches[batch_index]
        print("xdxd:")
        a = check(cur_batch)
        for ee in a:
            print(ee.eval())
        sess.run(optimizer, feed_dict={x: train_batches[batch_index]})
        if i % 100 == 0:
            # Calculate batch loss and accuracy
            l = sess.run([loss], feed_dict={x: train_batches[batch_index]})
            print("epoch " + str(i/100) + " loss: " + str(l))
        i += 1
