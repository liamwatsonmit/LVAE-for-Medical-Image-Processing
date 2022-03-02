""" import tensorflow.keras.backend as K
import tensorflow as tf

def dice_loss_per_frame(y_true, y_pred, smooth=0.0001):#y_pred.shape -> (batch,80,80,10)
    y_true=K.cast(y_true,'float32')
    y_pred = tf.reshape(y_pred, [10, 80, 80,80,10])
    union        = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    dice         = K.mean( (2. * intersection + smooth) / (union + smooth), axis=[0, 1])
    return 1-dice
def dice_loss_1(y_true, y_pred):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred)
def dice_loss_2(y_true, y_pred):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred)
def dice_loss_3(y_true, y_pred):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred)
def dice_loss_4(y_true, y_pred):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred)
def dice_loss_5(y_true, y_pred):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred)
def dice_loss_6(y_true, y_pred):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred)
def dice_loss_7(y_true, y_pred):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred)
def dice_loss_8(y_true, y_pred):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred)
def dice_loss_9(y_true, y_pred):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred)
def dice_loss_10(y_true, y_pred):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred)
dice_loss_every_frame = [dice_loss_1, dice_loss_2, dice_loss_3, dice_loss_4, dice_loss_5,
                         dice_loss_6, dice_loss_7, dice_loss_8, dice_loss_9, dice_loss_10]

def dice_loss_mean_3d(y_true, y_pred, smooth=0.0001):#y_pred.shape -> (batch,80,80,10)
    y_true=K.cast(y_true,'float32')
    union        = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    intersection = K.sum(y_true * y_pred, axis=[1,2, 3])
    dice         = K.mean( (2. * intersection + smooth) / (union + smooth), axis=[0,1])
    return 1-dice

def dice_loss_sum_3d(y_true, y_pred, smooth=0.0001):
    y_true=K.cast(y_true,'float32')
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    dice_loss    = K.sum(1 - (2. * intersection + smooth) / (union + smooth), axis=[0,1]) # dice < batch * 10 * 1
    return dice_loss

def kl_loss(inputs, gama=0.5, alpha=[0.00001, 0.0001, 0.01]):
    mu_e3, mu_p2, mu_q2, mu_p1, mu_q1, log_sqrt_sigma_e3, log_sqrt_sigma_p2, sigma_q2, log_sqrt_sigma_p1, sigma_q1 = inputs
    kl_loss_3 = -0.5 * K.sum(1 + log_sqrt_sigma_e3 - K.exp(log_sqrt_sigma_e3) - K.square(mu_e3), axis=-1)
    kl_loss_2 = 0.5 * K.sum(log_sqrt_sigma_p2 - 2 * K.log(sigma_q2) + (K.exp(log_sqrt_sigma_p2) + K.square(mu_p2 - mu_q2)) / (sigma_q2) - 1, axis=-1)
    kl_loss_1 = 0.5 * K.sum(log_sqrt_sigma_p1 - 2 * K.log(sigma_q1) + (K.exp(log_sqrt_sigma_p1) + K.square(mu_p1 - mu_q1)) / (sigma_q1) - 1, axis=-1)
    kl_loss_sum = gama * (alpha[2] * K.mean(kl_loss_3) + alpha[1] * K.mean(kl_loss_2) + alpha[0] * K.mean(kl_loss_1))
    return kl_loss_3, kl_loss_2, kl_loss_1, kl_loss_sum """

import tensorflow.keras.backend as K
import tensorflow as tf
def dice_loss_per_frame(y_true, y_pred, smooth=0.0001,index=0):#y_pred.shape -> (batch,80,80,10)
    y_true=K.cast(y_true,'float32')[...,index]
    y_pred = tf.reshape(y_pred, [10, 80, 80,80,10])
    y_pred=y_pred[...,index]
    union        = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    dice         = K.mean( (2. * intersection + smooth) / (union + smooth), axis=[0])
    return 1-dice

def dice_loss_sum_3d(y_true, y_pred, smooth=0.0001):
    y_true=K.cast(y_true,'float32')
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    dice_loss    = K.sum(1 - (2. * intersection + smooth) / (union + smooth), axis=[0,1]) # dice < batch * 10 * 1
    return dice_loss
    
def dice_loss_1(y_true, y_pred,index=0):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
def dice_loss_2(y_true, y_pred,index=1):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
def dice_loss_3(y_true, y_pred,index=2):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
def dice_loss_4(y_true, y_pred,index=3):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
def dice_loss_5(y_true, y_pred,index=4):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
def dice_loss_6(y_true, y_pred,index=5):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
def dice_loss_7(y_true, y_pred,index=6):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
def dice_loss_8(y_true, y_pred,index=7):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
def dice_loss_9(y_true, y_pred,index=8):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
def dice_loss_10(y_true, y_pred,index=9):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
dice_loss_every_frame = [dice_loss_1, dice_loss_2, dice_loss_3, dice_loss_4, dice_loss_5,
                         dice_loss_6, dice_loss_7, dice_loss_8, dice_loss_9, dice_loss_10]