import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import variable_scope
from .common_helpers import *
from .module import *
import horovod.tensorflow as hvd

def createGeneratorLoss(opt, label, logit, weight=None, fake=None, real=None):
    #cast the logits to fp32
    logit = ensure_type(logit, tf.float32)
    loss = None
    loss_type = opt['type']
    with_gan = opt['gan_loss']

    if loss_type == "weighted":
        #cast weights to FP32
        w_cast = ensure_type(weight, tf.float32)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=label,
                                                    logits=logit,
                                                    weights=w_cast,
                                                    reduction=tf.losses.Reduction.SUM)

    elif loss_type == "weighted_mean":
        #cast weights to FP32
        w_cast = ensure_type(weight, tf.float32)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=label,
                                                    logits=logit,
                                                    weights=w_cast,
                                                    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

    elif loss_type == "focal":
        #one-hot-encode
        labels_one_hot = tf.contrib.layers.one_hot_encoding(label, 3)
        #cast to FP32
        labels_one_hot = ensure_type(labels_one_hot, tf.float32)
        loss = focal_loss(onehot_labels=labels_one_hot, logits=logit, alpha=1., gamma=2.)
    
    if with_gan:
        real_logit = (real - tf.reduce_mean(fake))
        fake_logit = (fake - tf.reduce_mean(real))
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real_logit))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake_logit))
        loss = loss + opt['gan_weight'] * (real_loss + fake_loss)
    
    return loss

def createDiscriminatorLoss(opt, fake=None, real=None):
    loss = None
    with_gan = opt['gan_loss']

    if with_gan:
        fake_logit = (fake - tf.reduce_mean(real))
        real_logit = (real - tf.reduce_mean(fake))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake_logit))
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(real), logits=real_logit))
        loss = fake_loss + real_loss
        
        with_gan_gp = opt['gan_gp']
        # TODO
        if with_gan_gp:
            alpha = tf.random_uniform(shape=[real.shape[0], 1, 1, 1], minval=0., maxval=1.)
            interpolated = alpha*real + (1.-alpha)*fake
    
    return loss