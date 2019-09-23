from .networks import create
from .loss import createGeneratorLoss, createDiscriminatorLoss

def createModel(in_, opt):
    return create(in_, opt)

def createLoss(opt, logit, next_elem, fake=None, real=None):
    loss_opt = opt['Loss']
    generator_loss = createGeneratorLoss(loss_opt, next_elem[1], logit, next_elem[2], fake, real)
    discriminator_loss = createDiscriminatorLoss(loss_opt, fake, real)
    return generator_loss, discriminator_loss