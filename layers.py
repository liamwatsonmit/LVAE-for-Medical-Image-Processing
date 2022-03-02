import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras

def init(slices):

    inputs =  keras.layers.Input(shape=[80,80,10], name='cnn_inputs')
    inputs3d =  keras.layers.Input(shape=[80,80,80,slices], name='cnn3d_inputs')

    return inputs3d

class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var/2) + mean

class gen_mu_sigma_d(keras.layers.Layer):
    def call(self, inputs):
        log_sigma_e2, log_sigma_p2, mu_e, mu_p = inputs
        sigma_d = K.pow(K.exp(-log_sigma_e2) + K.exp(-log_sigma_p2), -1)
        mu_d = sigma_d * (mu_e * K.exp(-log_sigma_e2) + mu_p * K.exp(-log_sigma_p2))
        # return tf.clip_by_value(sigma_d, 1e-8, 50), mu_d
        return sigma_d,mu_d

class kl_loss_layer(keras.layers.Layer):
    def call(self, inputs):
        mu_e3,mu_p2, mu_q2, mu_p1, mu_q1,log_sqrt_sigma_e3,log_sqrt_sigma_p2, sigma_q2, log_sqrt_sigma_p1, sigma_q1 = inputs
        kl_loss_3 = -0.5 * K.sum(1 + log_sqrt_sigma_e3 - K.exp(log_sqrt_sigma_e3) - K.square(mu_e3), axis=-1)
        kl_loss_2 = 0.5 * K.sum(log_sqrt_sigma_p2 - 2 * K.log(sigma_q2) + (K.exp(log_sqrt_sigma_p2) + K.square(mu_p2 - mu_q2)) / (sigma_q2) - 1, axis=-1)
        kl_loss_1 = 0.5 * K.sum(log_sqrt_sigma_p1 - 2 * K.log(sigma_q1) + (K.exp(log_sqrt_sigma_p1) + K.square(mu_p1 - mu_q1)) / (sigma_q1) - 1, axis=-1)
        kl_loss_sum = 0.01 * K.mean(kl_loss_3) +0.001 * K.mean(kl_loss_2) +0.001 * K.mean(kl_loss_1)
        #self.gamma = tf.clip_by_value(self.gamma+0.1, 0.5, 100)
        # axis=[0, 1]
        print(kl_loss_3.shape, kl_loss_2.shape, kl_loss_1.shape, kl_loss_sum.shape)
        return kl_loss_3, kl_loss_2, kl_loss_1, kl_loss_sum*0.5
