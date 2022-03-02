from tensorflow import keras
from models.layers import Sampling, gen_mu_sigma_d, kl_loss_layer
import tensorflow.keras.backend as K

def lvae_encoder(z1_size=64,
                 z2_size=48,
                 z3_size=32):
    # lvae encoder
    lvae_encoder_inputs = keras.layers.Input(shape=[250], name='lvae_encoder_inputs')
    h1 = keras.layers.Dense(96, activation='elu', kernel_initializer=keras.initializers.RandomNormal(0, 0.02))(lvae_encoder_inputs)
    h1 = keras.layers.BatchNormalization(trainable=True,name='h1')(h1)
    h2 = keras.layers.Dense(48, activation='elu', kernel_initializer=keras.initializers.RandomNormal(0, 0.02))(h1)
    h2 = keras.layers.BatchNormalization(trainable=True)(h2)
    h2 = keras.layers.Dense(48, activation='elu', kernel_initializer=keras.initializers.RandomNormal(0, 0.02))(h2)
    h2 = keras.layers.BatchNormalization(trainable=True, name='h2')(h2)
    h3 = keras.layers.Dense(24, activation='elu', kernel_initializer=keras.initializers.RandomNormal(0, 0.02))(h2)
    h3 = keras.layers.BatchNormalization(trainable=True)(h3)
    h3 = keras.layers.Dense(12, activation='tanh', kernel_initializer=keras.initializers.RandomNormal(0, 0.02))(h3)
    h3 = keras.layers.BatchNormalization(trainable=True, name='h3')(h3)
    mu_e1 = keras.layers.Dense(z1_size, kernel_initializer='zeros', name='mu_e1')(h1)  # mu
    mu_e2 = keras.layers.Dense(z2_size, kernel_initializer='zeros', name='mu_e2')(h2)  # mu
    mu_e3 = keras.layers.Dense(z3_size, kernel_initializer='zeros', name='mu_e3')(h3)  # mu
    log_sqrt_sigma_e1 = keras.layers.Dense(z1_size, kernel_initializer='zeros', name='log_sqrt_sigma_e1sq')(h1)  # 2log_var
    log_sqrt_sigma_e2 = keras.layers.Dense(z2_size, kernel_initializer='zeros', name='log_sqrt_sigma_e2_sq')(
        h2)  # 2log_var
    log_sqrt_sigma_e3 = keras.layers.Dense(z3_size, kernel_initializer='zeros', name='log_sqrt_sigma_e3sq')(h3)  # 2log_var
    z3 = Sampling(name='z3')([mu_e3, log_sqrt_sigma_e3])
    encoder = keras.Model(inputs=[lvae_encoder_inputs],
                          outputs=[z3,
                                   mu_e1, mu_e2, mu_e3,
                                   log_sqrt_sigma_e1, log_sqrt_sigma_e2, log_sqrt_sigma_e3],
                          name='lvae_encoder')
    return encoder


def lvae_decoder(z1_size=64,
                 z2_size=48,
                 z3_size=32,
                 add_kl_loss=False):
    #inputs
    z3 = keras.layers.Input(shape=[z3_size],name='z3')
    mu_e1 = keras.layers.Input(shape=[z1_size], name='mu_e1')
    mu_e2 = keras.layers.Input(shape=[z2_size], name='mu_e2')
    mu_e3 = keras.layers.Input(shape=[z3_size], name='mu_e3')
    log_sqrt_sigma_e1 = keras.layers.Input(shape=[z1_size], name='log_sqrt_sigma_e1')
    log_sqrt_sigma_e2 = keras.layers.Input(shape=[z2_size], name='log_sqrt_sigma_e2')
    log_sqrt_sigma_e3 = keras.layers.Input(shape=[z3_size], name='log_sqrt_sigma_e3')

    # natwork
    x3 = keras.layers.Dense(24, activation='selu', kernel_initializer=keras.initializers.RandomNormal(0, 0.02))(z3)
    x3 = keras.layers.BatchNormalization(trainable=True)(x3)
    x3 = keras.layers.Dense(24, activation='selu', kernel_initializer=keras.initializers.RandomNormal(0, 0.02))(x3)
    x3 = keras.layers.BatchNormalization(trainable=True, name='x3')(x3)
    mu_p2               = keras.layers.Dense(z2_size, kernel_initializer='zeros', name='mu_p2')(x3)  # mu
    log_sqrt_sigma_p2   = keras.layers.Dense(z2_size, kernel_initializer='zeros', name='log_sqrt_sigma_p2')(x3)  # 2log_var
    sigma_q2, mu_q2     = gen_mu_sigma_d(name='sigma_q2_and_mu_q2')([log_sqrt_sigma_e2, log_sqrt_sigma_p2, mu_e2, mu_p2])
    z2                  = Sampling(name='z2')([mu_q2, 2. * K.log(sigma_q2)])
    x2 = keras.layers.Dense(48, activation='selu', kernel_initializer=keras.initializers.RandomNormal(0, 0.02))(z2)
    x2 = keras.layers.BatchNormalization(trainable=True)(x2)
    x2 = keras.layers.Dense(48, activation='selu', kernel_initializer=keras.initializers.RandomNormal(0, 0.02))(x2)
    x2 = keras.layers.BatchNormalization(trainable=True, name='x2')(x2)
    mu_p1               = keras.layers.Dense(z1_size, kernel_initializer='zeros', name='mu_p1')(x2)  # mu
    log_sqrt_sigma_p1   = keras.layers.Dense(z1_size, kernel_initializer='zeros', name='log_sqrt_sigma_p1')(x2)  # 2log_var
    sigma_q1, mu_q1     = gen_mu_sigma_d(name='sigma_q1_and_mu_q1')([log_sqrt_sigma_e1, log_sqrt_sigma_p1, mu_e1, mu_p1])
    z1                  = Sampling(name='z1')([mu_q1, 2. * K.log(sigma_q1)])
    output  = keras.layers.Dense(250, activation='selu', kernel_initializer=keras.initializers.RandomNormal(0, 0.02))(z1)
    output  = keras.layers.BatchNormalization(trainable=True, name='predicted')(output)
    decoder = keras.Model(inputs=[z3,
                                  mu_e1,mu_e2,mu_e3,
                                  log_sqrt_sigma_e1,log_sqrt_sigma_e2,log_sqrt_sigma_e3],
                       outputs=[output,
                                mu_e3,
                                mu_p2,mu_q2,
                                mu_p1,mu_q1,
                                log_sqrt_sigma_e3,
                                log_sqrt_sigma_p2,sigma_q2,
                                log_sqrt_sigma_p1,sigma_q1],
                       name='lvae_decoder')
    if add_kl_loss:
        kl_loss_3, kl_loss_2, kl_loss_1,kl_loss_sum = kl_loss_layer()([mu_e3,
                                                                       mu_p2, mu_q2,
                                                                       mu_p1, mu_q1,
                                                                       log_sqrt_sigma_e3,
                                                                       log_sqrt_sigma_p2,
                                                                       sigma_q2,
                                                                       log_sqrt_sigma_p1,
                                                                       sigma_q1])
        decoder.add_loss(kl_loss_sum)
    return decoder

def lvae_model(z1_size=64,
               z2_size=48,
               z3_size=32,
               add_kl_loss=False):
    encoder =   lvae_encoder(z1_size=z1_size, z2_size=z2_size, z3_size=z3_size)
    decoder =   lvae_decoder(z1_size=z1_size, z2_size=z2_size, z3_size=z3_size)
    lvae_inputs = keras.layers.Input(shape=[250], name='lvae_inputs')
    output, mu_e3, mu_p2, mu_q2, mu_p1, mu_q1, log_sqrt_sigma_e3, log_sqrt_sigma_p2, sigma_q2, log_sqrt_sigma_p1, sigma_q1 = decoder(encoder(lvae_inputs))
    if add_kl_loss:
        # loss
        _, _, _, kl_loss_sum = kl_loss_layer(name='kl_losses')([mu_e3,
                                                                mu_p2, mu_q2,
                                                                mu_p1, mu_q1,
                                                                log_sqrt_sigma_e3,
                                                                log_sqrt_sigma_p2,
                                                                sigma_q2,
                                                                log_sqrt_sigma_p1,
                                                                sigma_q1])
        #lvae.add_loss(kl_loss_sum)
        lvae = keras.Model(inputs=[lvae_inputs], outputs=[output,kl_loss_sum], name='lvae')
    else:
        lvae = keras.Model(inputs=[lvae_inputs], outputs=[output], name='lvae')
    return lvae

if __name__ == '__main__':
    # encoder = lvae_encoder()
    # encoder.summary()
    # decoder = lvae_decoder()
    # decoder.summary()
    model = lvae_model(add_kl_loss=True)
    model.summary()
    pass